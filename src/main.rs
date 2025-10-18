use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use ordered_float::OrderedFloat;
use regex::Regex;
use scraper::{Html as ScraperHtml, Selector};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs,
    net::SocketAddr,
    process::Command,
    sync::Arc,
};
use tempfile::tempdir;
use tokio::sync::RwLock;
use tokio::time::{sleep, Duration};
use url::{Position, Url};
use anyhow::{anyhow, Result};

// ================= CLI =================
#[derive(Parser, Debug, Clone)]
#[command(name = "site_qa", version, about = "Any-link Q&A with hybrid RAG (Ollama)")]
struct Cli {
    #[arg(long, env = "BIND_ADDR", default_value = "127.0.0.1:3000")]
    bind: String,

    #[arg(long, env = "OLLAMA_HOST", default_value = "http://localhost:11434")]
    ollama_host: String,

    #[arg(long, env = "EMBED_MODEL", default_value = "nomic-embed-text")]
    embed_model: String,

    #[arg(long, env = "GEN_MODEL", default_value = "llama3.1:8b")]
    gen_model: String,
}

// ================= Data =================
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Chunk {
    id: String,
    url: String,
    text: String,
    embedding: Vec<f32>,
    tf: HashMap<String, u32>,
    tok_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexFile {
    embed_model: String,
    gen_model: String,
    chunks: Vec<Chunk>,
    created_at: String,
    source_scope: String,
    df: HashMap<String, u32>,
    total_docs: usize,
    avg_len: f32,
}

#[derive(Clone)]
struct AppState {
    ollama_host: String,
    embed_model: String,
    gen_model: String,
    index: Arc<RwLock<Option<IndexFile>>>,
}

// ================= Utils =================
fn sanitize_url(raw: &str) -> Result<Url> {
    let token = raw.split_whitespace().next().unwrap_or("").trim();
    if token.is_empty() {
        return Err(anyhow!("Invalid URL: empty"));
    }
    let mut s = token.to_string();

    // strip control chars and surrounding quotes/brackets
    s.retain(|c| !c.is_control());
    s = s
        .trim_matches(|c: char| {
            matches!(c, '“'|'”'|'„'|'«'|'»'|'"'|'\''
                |'<'|'>'|'('|')'|'['|']'|'{'|'}')
        })
        .to_string();

    // normalize leading // or www.
    if s.starts_with("//") {
        s = format!("https:{s}");
    } else if s.to_ascii_lowercase().starts_with("www.") {
        s = format!("https://{s}");
    }

    if !(s.starts_with("http://") || s.starts_with("https://")) {
        s = format!("https://{}", s.trim_start_matches("://"));
    }
    Url::parse(&s).map_err(|e| anyhow!("Invalid URL: {}", e))
}

fn normalize_whitespace(s: &str) -> String {
    let ws = Regex::new(r"\s+").unwrap();
    ws.replace_all(s, " ").trim().to_string()
}

fn strip_url_fragment(u: &Url) -> String {
    let mut s = u[..Position::AfterPath].to_string();
    if let Some(q) = u.query() {
        s.push('?');
        s.push_str(q);
    }
    s
}

fn chunk_text(text: &str, target: usize, overlap: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![];
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let end = (start + target).min(chars.len());
        let slice: String = chars[start..end].iter().collect();
        out.push(slice);
        if end == chars.len() { break; }
        start = end.saturating_sub(overlap);
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

// UTF-8 safe clamps (use byte indices)
fn clamp_for_embedding(s: &str) -> String {
    let max_chars: usize = std::env::var("EMBED_MAX_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(750);

    let mut count = 0usize;
    let mut last_space_byte: Option<usize> = None;
    let mut last_byte: Option<usize> = None;

    for (byte_idx, ch) in s.char_indices() {
        last_byte = Some(byte_idx);
        if ch.is_whitespace() {
            last_space_byte = Some(byte_idx);
        }
        count += 1;
        if count >= max_chars {
            let cut = last_space_byte.or(last_byte).unwrap_or(s.len());
            return s[..cut].to_string();
        }
    }
    s.to_string()
}

fn clamp_to(s: &str, max_chars: usize) -> String {
    let mut count = 0usize;
    let mut last_space_byte: Option<usize> = None;
    let mut last_byte: Option<usize> = None;

    for (byte_idx, ch) in s.char_indices() {
        last_byte = Some(byte_idx);
        if ch.is_whitespace() {
            last_space_byte = Some(byte_idx);
        }
        count += 1;
        if count >= max_chars {
            let cut = last_space_byte.or(last_byte).unwrap_or(s.len());
            return s[..cut].to_string();
        }
    }
    s.to_string()
}

fn embed_chunk_size() -> usize {
    // configurable: CHUNK_TARGET_CHARS (default 700)
    std::env::var("CHUNK_TARGET_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(700)
}

// ================= HTTP client =================
async fn build_http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .pool_idle_timeout(Some(Duration::from_secs(30)))
        .timeout(Duration::from_secs(45))
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36")
        .build()?)
}

async fn fetch_html_with_retries(client: &reqwest::Client, url: &Url, referer: Option<&str>) -> Result<String> {
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=3 {
        let mut req = client.get(url.clone())
            .header(reqwest::header::ACCEPT, "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9,de;q=0.7");
        if let Some(r) = referer { req = req.header(reqwest::header::REFERER, r); }
        match req.send().await {
            Ok(resp) => match resp.error_for_status() {
                Ok(ok) => match ok.text().await {
                    Ok(t) => return Ok(t),
                    Err(e) => last_err = Some(e.into()),
                },
                Err(e) => last_err = Some(e.into()),
            },
            Err(e) => last_err = Some(e.into()),
        }
        sleep(Duration::from_millis(200 * attempt as u64)).await;
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown fetch error")))
}

async fn fetch_bytes_with_retries(client: &reqwest::Client, url: &Url, referer: Option<&str>) -> Result<Vec<u8>> {
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=3 {
        let mut req = client.get(url.clone())
            .header(reqwest::header::ACCEPT, "application/pdf,application/octet-stream,*/*")
            .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9,de;q=0.7");
        if let Some(r) = referer { req = req.header(reqwest::header::REFERER, r); }
        match req.send().await {
            Ok(resp) => match resp.error_for_status() {
                Ok(ok) => match ok.bytes().await {
                    Ok(b) => return Ok(b.to_vec()),
                    Err(e) => last_err = Some(e.into()),
                },
                Err(e) => last_err = Some(e.into()),
            },
            Err(e) => last_err = Some(e.into()),
        }
        sleep(Duration::from_millis(200 * attempt as u64)).await;
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown fetch error")))
}

// ================= Scraping =================
fn looks_like_pdf(url: &Url) -> bool {
    let s = url.as_str().to_ascii_lowercase();
    s.ends_with(".pdf") || s.contains(".pdf?")
}

fn pdf_bytes_to_text(pdf: &[u8]) -> Result<String> {
    // Optional: limit pages for speed via env PDF_MAX_PAGES (default 12)
    let max_pages: usize = std::env::var("PDF_MAX_PAGES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);

    let dir = tempdir()?;
    let in_path = dir.path().join("doc.pdf");
    let out_path = dir.path().join("doc.txt");
    fs::write(&in_path, pdf)?;
    // -q (quiet) hides warnings; -f/-l restrict pages for speed
    let status = Command::new("pdftotext")
        .args([
            "-q",
            "-layout",
            "-enc", "UTF-8",
            "-f", "1",
            "-l", &max_pages.to_string(),
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        anyhow::bail!("pdftotext exited with non-zero status");
    }
    let txt = fs::read_to_string(out_path)?;
    Ok(normalize_whitespace(&txt))
}

fn extract_text_and_links(base: &Url, html: &str) -> (String, Vec<Url>) {
    let doc = ScraperHtml::parse_document(html);
    let mut text_buf = String::new();
    for sel in &["main", "article", "body"] {
        if let Ok(s) = Selector::parse(sel) {
            if let Some(node) = doc.select(&s).next() {
                for t in node.text() {
                    let t = normalize_whitespace(t);
                    if !t.is_empty() {
                        text_buf.push_str(&t);
                        text_buf.push(' ');
                    }
                }
                break;
            }
        }
    }
    let a_sel = Selector::parse("a[href]").unwrap();
    let mut links = Vec::new();
    for a in doc.select(&a_sel) {
        if let Some(href) = a.value().attr("href") {
            if let Ok(abs) = base.join(href) {
                links.push(abs);
            }
        }
    }
    (normalize_whitespace(&text_buf), links)
}

// ================= Crawl =================
async fn crawl(start: &Url, depth: usize, scope_prefix: &str, max_pages: usize) -> Result<Vec<(String, String)>> {
    let client = build_http_client().await?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut q: VecDeque<(Url, usize, Option<String>)> = VecDeque::new();
    q.push_back((start.clone(), 0, None));

    let bar = ProgressBar::new(max_pages as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] {wide_bar} {pos}/{len} {msg}")
            .unwrap(),
    );

    let skip_pdfs = std::env::var("SKIP_PDFS").ok().as_deref() == Some("1");

    while let Some((u, d, referer)) = q.pop_front() {
        if out.len() >= max_pages { break; }
        let canonical = strip_url_fragment(&u);
        if !seen.insert(canonical.clone()) { continue; }

        match fetch_html_with_retries(&client, &u, referer.as_deref()).await {
            Ok(html) => {
                let (text, links) = extract_text_and_links(&u, &html);
                if !text.trim().is_empty() {
                    out.push((canonical.clone(), text));
                }
                bar.inc(1);

                if d < depth {
                    for link in links {
                        let link_key = strip_url_fragment(&link);

                        if looks_like_pdf(&link) {
                            if skip_pdfs { continue; }
                            if link.origin() != start.origin() { continue; } // keep to same origin
                            if seen.insert(link_key.clone()) {
                                if let Ok(bytes) = fetch_bytes_with_retries(&client, &link, Some(u.as_str())).await {
                                    // skip very large PDFs for speed
                                    if bytes.len() > 10 * 1024 * 1024 {
                                        continue;
                                    }
                                    if let Ok(txt) = pdf_bytes_to_text(&bytes) {
                                        if !txt.trim().is_empty() {
                                            out.push((link_key.clone(), txt));
                                            bar.inc(1);
                                        }
                                    }
                                }
                            }
                        } else {
                            if link_key.starts_with(scope_prefix) {
                                q.push_back((link, d + 1, Some(u.as_str().to_string())));
                            }
                        }
                    }
                }
            }
            Err(_) => { /* ignore page fetch errors */ }
        }
        // politeness delay
        sleep(Duration::from_millis(200)).await;
    }

    bar.finish_and_clear();
    Ok(out)
}

// ================= Ollama API =================
#[derive(Serialize)]
struct EmbeddingsReq<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
}
#[derive(Deserialize)]
struct EmbeddingsResp { embedding: Vec<f32> }

async fn embed_text(ollama: &str, model: &str, text: &str) -> Result<Vec<f32>> {
    // For debugging: allow disabling embeddings
    if std::env::var("DISABLE_EMBEDDINGS").ok().as_deref() == Some("1") {
        return Ok(Vec::new());
    }

    // UTF-8 safe clamp + smaller context by env
    let mut safe = clamp_for_embedding(text);
    let mut num_ctx: usize = std::env::var("EMBED_NUM_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);

    // Retry path: progressively shrink if context still too long
    let mut tries = 0usize;
    loop {
        let resp = reqwest::Client::new()
            .post(format!("{}/api/embeddings", ollama))
            .json(&EmbeddingsReq {
                model,
                prompt: &safe,
                options: Some(serde_json::json!({
                    "num_ctx": num_ctx,
                    "truncate": true
                })),
            })
            .send()
            .await?;

        let status = resp.status();
        if status.is_success() {
            let data = resp.json::<EmbeddingsResp>().await?;
            return Ok(data.embedding);
        } else {
            let body = resp.text().await.unwrap_or_default();
            let lower = body.to_ascii_lowercase();
            let is_ctx = lower.contains("context length")
                || lower.contains("exceeds the context length")
                || lower.contains("too long");
            if is_ctx && tries < 4 {
                // shrink harder each try
                let new_limit = match tries {
                    0 => 256,
                    1 => 192,
                    2 => 160,
                    _ => 120,
                };
                safe = clamp_to(text, new_limit);
                num_ctx = num_ctx.min(1024);
                tries += 1;
                continue;
            }
            anyhow::bail!(
                "Embeddings failed ({}): {}. Hint: lower CHUNK_TARGET_CHARS/EMBED_MAX_CHARS or pick a bigger-context embedding model.",
                status, body
            );
        }
    }
}

#[derive(Serialize)]
struct GenerateReq<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}
#[derive(Deserialize)]
struct GenerateChunk { response: Option<String> }

async fn generate(ollama: &str, model: &str, prompt: &str, temperature: f32) -> Result<String> {
    let mut res = reqwest::Client::new()
        .post(format!("{}/api/generate", ollama))
        .json(&GenerateReq { model, prompt, temperature: Some(sanitize_temp(temperature)), stream: true })
        .send().await?
        .error_for_status()?;

    let mut out = String::new();
    while let Some(chunk) = res.chunk().await? {
        let line = String::from_utf8_lossy(&chunk).to_string();
        for part in line.lines() {
            if part.trim().is_empty() { continue; }
            if let Ok(tick) = serde_json::from_str::<GenerateChunk>(part) {
                if let Some(s) = tick.response { out.push_str(&s); }
            }
        }
    }
    Ok(out)
}

fn sanitize_temp(t: f32) -> f32 {
    if t.is_nan() { 0.2 } else { t.clamp(0.0, 1.0) }
}

// ================= Lexical & BM25 =================
fn tokenize_lower(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

fn bow_tf(tokens: &[String]) -> HashMap<String, u32> {
    let mut m = HashMap::new();
    for t in tokens {
        *m.entry(t.clone()).or_insert(0) += 1;
    }
    m
}

fn expand_query_terms(q: &str) -> Vec<String> {
    let mut terms: HashSet<String> = tokenize_lower(q).into_iter().collect();
    let ql = q.to_ascii_lowercase();

    if ql.contains("incharge") || ql.contains("in charge") {
        terms.insert("responsible".into());
        terms.insert("contact".into());
        terms.insert("head".into());
    }
    if ql.contains("admission") || ql.contains("admissions") {
        terms.insert("enrolment".into());
        terms.insert("studierendensekretariat".into());
        terms.insert("admissions".into());
        terms.insert("admissions office".into());
    }
    if ql.contains("uniassist") || ql.contains("uni-assist") || ql.contains("uni assist") {
        terms.insert("uni-assist".into());
        terms.insert("uni".into());
        terms.insert("assist".into());
    }
    if ql.contains("aps") {
        terms.insert("akademische".into());
        terms.insert("prüfstelle".into());
    }
    if ql.contains("deadline") || ql.contains("last date") || ql.contains("closing date") {
        terms.insert("application".into());
        terms.insert("closing".into());
        terms.insert("date".into());
    }
    if ql.contains("ects") || ql.contains("credit") || ql.contains("credits") || ql.contains("points") {
        terms.insert("ects".into());
        terms.insert("credit".into());
        terms.insert("module".into());
        terms.insert("thesis".into());
    }
    terms.into_iter().collect()
}

fn bm25_score(
    q_terms: &[String],
    chunk: &Chunk,
    df: &HashMap<String, u32>,
    total_docs: usize,
    avg_len: f32,
) -> f32 {
    if total_docs == 0 || avg_len == 0.0 { return 0.0; }
    let k1 = 1.5_f32;
    let b  = 0.75_f32;

    let mut score = 0.0_f32;
    for term in q_terms {
        let f = *chunk.tf.get(term).unwrap_or(&0) as f32;
        if f <= 0.0 { continue; }
        let df_t = *df.get(term).unwrap_or(&0) as f32;
        if df_t <= 0.0 { continue; }
        let idf = ((total_docs as f32 - df_t + 0.5) / (df_t + 0.5) + 1e-6).ln();
        let denom = f + k1 * (1.0 - b + b * (chunk.tok_len as f32 / avg_len));
        score += idf * (f * (k1 + 1.0) / denom);
    }
    score
}

// ================= Index & Hybrid RAG =================
async fn build_index(
    ollama: &str,
    embed_model: &str,
    gen_model: &str,
    pairs: Vec<(String, String)>,
) -> Result<IndexFile> {
    let mut chunks = Vec::new();
    let mut df: HashMap<String, u32> = HashMap::new();
    let mut total_len: usize = 0;

    let target = embed_chunk_size();   // e.g., 700 (override via env)
    for (url, text) in pairs {
        for (i, piece) in chunk_text(&text, target, 120).into_iter().enumerate() {
            let tokens = tokenize_lower(&piece);
            let tf = bow_tf(&tokens);
            let tok_len = tokens.len();
            total_len += tok_len;

            let mut seen: HashSet<&String> = HashSet::new();
            for term in tf.keys() {
                if seen.insert(term) {
                    *df.entry(term.clone()).or_insert(0) += 1;
                }
            }

            let emb = embed_text(ollama, embed_model, &piece).await?;
            chunks.push(Chunk {
                id: format!("{}#{}", url, i),
                url: url.clone(),
                text: piece,
                embedding: emb,
                tf,
                tok_len,
            });
        }
    }

    let total_docs = chunks.len();
    let avg_len = if total_docs == 0 { 0.0 } else { total_len as f32 / total_docs as f32 };

    Ok(IndexFile {
        embed_model: embed_model.to_string(),
        gen_model: gen_model.to_string(),
        chunks,
        created_at: Utc::now().to_rfc3339(),
        source_scope: String::new(),
        df,
        total_docs,
        avg_len,
    })
}

fn keyword_bonus(text: &str, url: &str, q: &str) -> f32 {
    let t = text.to_ascii_lowercase();
    let u = url.to_ascii_lowercase();
    let ql = q.to_ascii_lowercase();
    let mut s: f32 = 0.0;

    for h in ["ects","credit","credits","thesis","module","modules","study plan","curriculum","program structure","pflichtbereich","wahlpflichtbereich"] {
        if ql.contains(h) && (t.contains(h) || u.contains(h)) { s += 0.9; }
    }
    for h in ["uni-assist","uni assist","aps","application deadline","admissions office","studierendensekretariat"] {
        if ql.contains(h) && (t.contains(h) || u.contains(h)) { s += 0.8; }
    }
    if (ql.contains("english") || ql.contains("program")) &&
        (t.contains("master of science") || t.contains("master of arts") || t.contains("master of laws") || t.contains("english-taught")) {
        s += 0.6;
    }
    if ql.contains("who") || ql.contains("incharge") || ql.contains("in charge") || ql.contains("responsible") || ql.contains("contact") {
        if Regex::new(r"[A-Z][a-z]+ [A-Z][a-z]+").unwrap().is_match(&t) { s += 0.5; }
        if t.contains('@') || Regex::new(r"room\s*\d+").unwrap().is_match(&t) { s += 0.3; }
    }
    let num_re = Regex::new(r"\b\d{1,3}\b").unwrap();
    for m in num_re.find_iter(&ql) {
        let num = m.as_str();
        if t.contains(num) || u.contains(num) { s += 0.2; }
    }
    s.min(2.0)
}

fn rerank_hybrid<'a>(
    question: &str,
    emb_q: &[f32],
    chunks: &'a [Chunk],
    df: &HashMap<String, u32>,
    total_docs: usize,
    avg_len: f32,
    take: usize,
) -> Vec<(&'a Chunk, f32)> {
    let q_terms = expand_query_terms(question);

    let mut prelim: Vec<(&Chunk, f32)> = chunks
        .iter()
        .map(|c| (c, cosine(emb_q, &c.embedding)))
        .collect();
    prelim.sort_by_key(|(_, s)| OrderedFloat(-*s));
    prelim.truncate(take.max(50));

    let mut scored: Vec<(&Chunk, f32)> = prelim
        .into_iter()
        .map(|(c, cos)| {
            let bm = bm25_score(&q_terms, c, df, total_docs, avg_len);
            let kb = keyword_bonus(&c.text, &c.url, question);
            let score = 0.55 * cos + 0.35 * bm + 0.10 * kb;
            (c, score)
        })
        .collect();

    scored.sort_by_key(|(_, s)| OrderedFloat(-*s));
    scored.truncate(take);
    scored
}

fn choose_primary_source(picks: &[(&Chunk, f32)]) -> String {
    if let Some((chunk, _)) = picks.first() {
        chunk.url.clone()
    } else {
        String::new()
    }
}

// ================= Comprehensive prompt (not “3 points”) =================
fn build_prompt(question: &str, contexts: &[(&Chunk, f32)], primary_source: &str) -> String {
    // Pack top contexts (already reranked)
    let mut ctx = String::new();
    for (c, _) in contexts {
        ctx.push_str(&format!("SOURCE URL: {}\n{}\n\n", c.url, c.text));
    }

    // Light intent flags for subtle guidance
    let ql = question.to_ascii_lowercase();
    let wants_list     = ql.contains("list") || ql.contains("which programs") || ql.contains("what programs");
    let wants_deadline = ql.contains("deadline") || ql.contains("last date") || ql.contains("closing date");
    let wants_contact  = ql.contains("who") || ql.contains("contact") || ql.contains("incharge") || ql.contains("in charge");
    let wants_require  = ql.contains("requirement") || ql.contains("eligibility") || ql.contains("admission") || ql.contains("uni-assist") || ql.contains("aps");

    let mut subtle_rules = String::new();
    if wants_list {
        subtle_rules.push_str("- If the question asks for a list, provide a complete bullet list using the exact titles/names found in CONTEXT.\n");
    }
    if wants_deadline {
        subtle_rules.push_str("- Give exact dates first (with semester labels if present), and specify the portal (e.g., uni-assist vs. university) if stated.\n");
    }
    if wants_contact {
        subtle_rules.push_str("- Include full contact details if present: name, role, office/room, email/phone.\n");
    }
    if wants_require {
        subtle_rules.push_str("- If requirements are present, include a clear checklist (degree, language level, uni-assist/APS, documents).\n");
    }

    let global_rules = r#"- Use ONLY the CONTEXT. Do NOT invent details.
- Quote exact numbers, dates, names and program titles.
- Prefer concise paragraphs and bullet points. Use short headings if helpful.
- Include short quotes only when needed to preserve exact wording.
- End with one source line:  Source: <URL>."#;

    let primary = if primary_source.is_empty() { "(unknown)" } else { primary_source };

    // Convert String -> &str for the formatter
    let subtle_rules_view: &str = if subtle_rules.is_empty() { "(none)" } else { &subtle_rules };

    format!(
r#"You are an expert university admissions/curriculum assistant.

{global_rules}

Additional guidance:
{subtle_rules}

QUESTION:
{q}

CONTEXT:
{ctx}

Write a comprehensive, precise answer strictly from the CONTEXT. Be complete (not a 3-point summary). Use clear paragraphs and bullets where helpful. End with:
Source: {primary}
"#,
        q = question,
        ctx = ctx,
        global_rules = global_rules,
        subtle_rules = subtle_rules_view,   // <- &str view fixes E0308
        primary = primary
    )
}


// ================= HTTP types =================
#[derive(Deserialize)]
struct IndexReq { url: String, depth: Option<usize>, max_pages: Option<usize>, scope_prefix: Option<String> }
#[derive(Serialize)]
struct IndexResp { ok: bool, chunks: usize, pages_indexed: usize, created_at: String, source_scope: String }
#[derive(Deserialize)]
struct AskReq {
    question: String,
    top_k: Option<usize>,
    temperature: Option<f32>,
    start_url: Option<String>,
    depth: Option<usize>,
    max_pages: Option<usize>,
    scope_prefix: Option<String>,
}
#[derive(Serialize)]
struct AskResp { answer: String, sources: Vec<String> }

// ================= Helpers for auto-index =================
fn same_origin(a: &Url, b: &Url) -> bool { a.origin() == b.origin() }

fn index_matches_link(idx: &IndexFile, start: &Url) -> bool {
    if idx.chunks.is_empty() { return false; }
    if let Ok(idx_first) = Url::parse(&idx.chunks[0].url) {
        return same_origin(&idx_first, start);
    }
    false
}

// ================= Handlers =================
async fn index_site(State(st): State<AppState>, Json(req): Json<IndexReq>) -> impl IntoResponse {
    let depth = req.depth.filter(|d| *d > 0).unwrap_or(4);
    let max_pages = req.max_pages.filter(|m| *m > 0).unwrap_or(400);

    let start = match sanitize_url(&req.url) {
        Ok(u) => u,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };

    // Default scope: scheme+host only (BeforePath).
    let scope = req
        .scope_prefix
        .unwrap_or_else(|| start[..Position::BeforePath].to_string());

    let pairs = match crawl(&start, depth, &scope, max_pages).await {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => return (StatusCode::BAD_REQUEST, "Crawl returned 0 pages".to_string()).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Crawl failed: {e:#}")).into_response(),
    };

    let mut idx = match build_index(&st.ollama_host, &st.embed_model, &st.gen_model, pairs).await {
        Ok(i) => i,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Index failed: {e:#}")).into_response(),
    };
    idx.source_scope = scope.clone();

    let resp = IndexResp {
        ok: true,
        chunks: idx.chunks.len(),
        pages_indexed: idx.chunks.iter().map(|c| &c.url).collect::<HashSet<_>>().len(),
        created_at: idx.created_at.clone(),
        source_scope: scope.clone(),
    };

    *st.index.write().await = Some(idx);

    Json(resp).into_response()
}

async fn ask(State(st): State<AppState>, Json(req): Json<AskReq>) -> impl IntoResponse {
    // Auto-index if start_url present and current index doesn't match
    if let Some(start_raw) = &req.start_url {
        let start_url = match sanitize_url(start_raw) {
            Ok(u) => u,
            Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid start_url: {}", e)).into_response(),
        };

        let need_reindex = {
            let guard = st.index.read().await;
            match &*guard {
                None => true,
                Some(idx) => !index_matches_link(idx, &start_url),
            }
        };

        if need_reindex {
            let depth = req.depth.filter(|d| *d > 0).unwrap_or(4);
            let max_pages = req.max_pages.filter(|m| *m > 0).unwrap_or(400);
            let scope = req
                .scope_prefix
                .clone()
                .unwrap_or_else(|| start_url[..url::Position::BeforePath].to_string());

            let pairs = match crawl(&start_url, depth, &scope, max_pages).await {
                Ok(p) if !p.is_empty() => p,
                Ok(_) => return (StatusCode::BAD_REQUEST, "Crawl returned 0 pages".to_string()).into_response(),
                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Crawl failed: {e:#}")).into_response(),
            };

            let mut idx = match build_index(&st.ollama_host, &st.embed_model, &st.gen_model, pairs).await {
                Ok(i) => i,
                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Index failed: {e:#}")).into_response(),
            };
            idx.source_scope = scope.clone();

            *st.index.write().await = Some(idx);
        }
    }

    // Answering
    let idx = match st.index.read().await.clone() {
        Some(i) => i,
        None => return (StatusCode::BAD_REQUEST, "No index loaded. Provide start_url or index a site first.".to_string()).into_response(),
    };

    let emb_q = match embed_text(&st.ollama_host, &idx.embed_model, &req.question).await {
        Ok(e) => e,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Embed failed: {e:#}")).into_response(),
    };

    let ql = req.question.to_ascii_lowercase();
    let list_programs = (ql.contains("english") || ql.contains("in english"))
        && (ql.contains("program") || ql.contains("study program") || ql.contains("list"));

    let default_k = if list_programs { 30 } else { 18 };
    let retrieval_k = req.top_k.unwrap_or(default_k);
    let picks = rerank_hybrid(
        &req.question, &emb_q, &idx.chunks, &idx.df, idx.total_docs, idx.avg_len,
        retrieval_k.min(default_k),
    );

    let primary_link = choose_primary_source(&picks);
    let prompt = build_prompt(&req.question, &picks, &primary_link);
    let temperature = if list_programs { 0.0 } else { req.temperature.unwrap_or(0.25) };

    let mut answer = match generate(&st.ollama_host, &idx.gen_model, &prompt, temperature).await {
        Ok(a) => a,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Generation failed: {e:#}")).into_response(),
    };

    let mut seen = HashSet::new();
    let mut sources: Vec<String> = Vec::new();
    if !primary_link.is_empty() && seen.insert(primary_link.clone()) {
        sources.push(primary_link.clone());
    }
    for (c, _) in &picks {
        if seen.insert(c.url.clone()) { sources.push(c.url.clone()); }
        if sources.len() >= 8 { break; }
    }

    if !answer.to_ascii_lowercase().contains("source:") {
        if let Some(first) = sources.first() {
            answer.push_str("\n\nSource: ");
            answer.push_str(first);
        }
    }

    Json(AskResp { answer, sources }).into_response()
}

// ================= Static HTML =================
async fn index_html() -> impl IntoResponse {
    Html(include_str!("../static/index.html"))
}

// ================= Main =================
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let state = AppState {
        ollama_host: cli.ollama_host,
        embed_model: cli.embed_model,
        gen_model: cli.gen_model,
        index: Arc::new(RwLock::new(None)),
    };

    let app = Router::new()
        .route("/", get(index_html))
        .route("/api/index", post(index_site))
        .route("/api/ask", post(ask))
        .with_state(state);

    let addr: SocketAddr = cli.bind.parse()?;
    println!("➡️  Open http://{addr}/");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

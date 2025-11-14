use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use regex::Regex;
use scraper::{Html as ScraperHtml, Selector};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs,
    hash::{Hash, Hasher},
    io::Write,
    net::SocketAddr,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
};
use tempfile::tempdir;
use tokio::sync::RwLock;
use tokio::time::{sleep, Duration};
use url::{Position, Url};

type Anyhow<T> = Result<T, anyhow::Error>;
use anyhow::{anyhow, bail, Context};

/// ================= CLI =================
#[derive(Parser, Debug, Clone)]
#[command(name = "site_qa", version, about = "Any-link/file Q&A with hybrid RAG (Ollama)")]
struct Cli {
    #[arg(long, env = "BIND_ADDR", default_value = "127.0.0.1:3000")]
    bind: String,

    #[arg(long, env = "OLLAMA_HOST", default_value = "http://localhost:11434")]
    ollama_host: String,

    #[arg(long, env = "EMBED_MODEL", default_value = "all-minilm")]
    embed_model: String,

    #[arg(long, env = "GEN_MODEL", default_value = "llama3.1:8b")]
    gen_model: String,
}

/// ================= Data =================
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Chunk {
    id: String,
    url: String, // logical source (URL or file://)
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
    df: HashMap<String, u32>, // document frequency over chunks
    total_docs: usize,
    avg_len: f32,
}

#[derive(Clone)]
struct AppState {
    ollama_host: String,
    embed_model: String,
    gen_model: String,
    // session_id -> index (in-memory)
    sessions: Arc<RwLock<HashMap<String, IndexFile>>>,
}

/// ================= Utils =================
fn sanitize_url(raw: &str) -> Anyhow<Url> {
    let token = raw.split_whitespace().next().unwrap_or("").trim();
    if token.is_empty() {
        bail!("Invalid URL: empty");
    }
    let mut s = token.to_string();

    // strip control chars and surrounding quotes/brackets
    s.retain(|c| !c.is_control());
    s = s
        .trim_matches(|c: char| {
            matches!(
                c,
                '“' | '”' | '„' | '«' | '»' | '"' | '\''
                    | '<' | '>' | '(' | ')' | '[' | ']' | '{' | '}'
            )
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

static WS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
fn normalize_ws(s: &str) -> String {
    WS.replace_all(s, " ").trim().to_string()
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
        if end == chars.len() {
            break;
        }
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
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

// UTF-8 safe clamps (char boundary aware via char_indices)
fn clamp_for_embedding(s: &str) -> String {
    let max_chars: usize = std::env::var("EMBED_MAX_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);

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
    std::env::var("CHUNK_TARGET_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// ================= HTTP client =================
async fn build_http_client() -> Anyhow<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .pool_idle_timeout(Some(Duration::from_secs(30)))
        .timeout(Duration::from_secs(45))
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36")
        .build()?)
}

async fn fetch_html(client: &reqwest::Client, url: &Url, referer: Option<&str>) -> Anyhow<String> {
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=3 {
        let mut req = client
            .get(url.clone())
            .header(
                reqwest::header::ACCEPT,
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9,de;q=0.7");
        if let Some(r) = referer {
            req = req.header(reqwest::header::REFERER, r);
        }
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
        sleep(Duration::from_millis(180 * attempt as u64)).await;
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown fetch error")))
}

async fn fetch_bytes(client: &reqwest::Client, url: &Url, referer: Option<&str>) -> Anyhow<Vec<u8>> {
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=3 {
        let mut req = client
            .get(url.clone())
            .header(
                reqwest::header::ACCEPT,
                "application/pdf,application/octet-stream,*/*",
            )
            .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9,de;q=0.7");
        if let Some(r) = referer {
            req = req.header(reqwest::header::REFERER, r);
        }
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
        sleep(Duration::from_millis(180 * attempt as u64)).await;
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown fetch error")))
}

/// ================= Scraping =================
fn looks_like_pdf(url: &Url) -> bool {
    let s = url.as_str().to_ascii_lowercase();
    s.ends_with(".pdf") || s.contains(".pdf?")
}

fn extract_text_and_links(base: &Url, html: &str) -> (String, Vec<Url>) {
    let doc = ScraperHtml::parse_document(html);
    let mut text_buf = String::new();

    for sel in &["main", "article", "body"] {
        if let Ok(s) = Selector::parse(sel) {
            if let Some(node) = doc.select(&s).next() {
                for t in node.text() {
                    let t = normalize_ws(t);
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
    (normalize_ws(&text_buf), links)
}

fn have_cmd(name: &str) -> bool {
    which::which(name).is_ok()
}

fn pdf_bytes_to_text(pdf: &[u8]) -> Anyhow<String> {
    // Try pdftotext first
    if have_cmd("pdftotext") {
        let max_pages: usize = std::env::var("PDF_MAX_PAGES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);

        let dir = tempdir()?;
        let in_path = dir.path().join("doc.pdf");
        let out_path = dir.path().join("doc.txt");
        fs::write(&in_path, pdf)?;
        let status = Command::new("pdftotext")
            .args([
                "-q", "-layout", "-enc", "UTF-8", "-f", "1", "-l", &max_pages.to_string(),
                in_path.to_str().unwrap(),
                out_path.to_str().unwrap(),
            ])
            .status()?;
        if !status.success() {
            bail!("pdftotext exited with non-zero status");
        }
        let txt = fs::read_to_string(out_path)?;
        return Ok(normalize_ws(&txt));
    }

    // Fallback: try Python pypdf (best-effort)
    let dir = tempdir()?;
    let in_path = dir.path().join("doc.pdf");
    fs::write(&in_path, pdf)?;
    let code = r#"
import sys
from pypdf import PdfReader
p=PdfReader(sys.argv[1])
out=[]
for i,pg in enumerate(p.pages):
    if i>20: break
    try: out.append(pg.extract_text() or "")
    except: pass
print("\n".join(out))
"#;
    let py = which::which("python3")
        .or_else(|_| which::which("python"))
        .context("No python found; install poppler's pdftotext or python+pypdf")?;
    let out = Command::new(py)
        .arg("-c")
        .arg(code)
        .arg(in_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()?;
    let txt = String::from_utf8_lossy(&out.stdout).to_string();
    Ok(normalize_ws(&txt))
}

/// ================= Crawl =================
async fn crawl(
    start: &Url,
    depth: usize,
    scope_prefix: &str,
    max_pages: usize,
) -> Anyhow<Vec<(String, String)>> {
    let client = build_http_client().await?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut q: VecDeque<(Url, usize, Option<String>)> = VecDeque::new();
    q.push_back((start.clone(), 0, None));

    let per_page_link_cap: usize = std::env::var("MAX_LINKS_PER_PAGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);

    let bar = ProgressBar::new(max_pages as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] {wide_bar} {pos}/{len} {msg}")
            .unwrap(),
    );

    let allow_pdfs = std::env::var("ALLOW_PDFS").ok().as_deref() == Some("1");
    let crawl_delay_ms = env_u64("CRAWL_DELAY_MS", 120);

    while let Some((u, d, referer)) = q.pop_front() {
        if out.len() >= max_pages {
            break;
        }
        let canonical = strip_url_fragment(&u);
        if !seen.insert(canonical.clone()) {
            continue;
        }

        match fetch_html(&client, &u, referer.as_deref()).await {
            Ok(html) => {
                let (text, all_links) = extract_text_and_links(&u, &html);
                if !text.trim().is_empty() {
                    out.push((canonical.clone(), text));
                }
                bar.inc(1);

                if d < depth {
                    let mut added = 0usize;
                    for link in all_links {
                        if added >= per_page_link_cap {
                            break;
                        }
                        let link_key = strip_url_fragment(&link);

                        if looks_like_pdf(&link) {
                            if !allow_pdfs {
                                continue;
                            }
                            if link.origin() != start.origin() {
                                continue; // stay on origin for PDFs
                            }
                            if seen.insert(link_key.clone()) {
                                if let Ok(bytes) =
                                    fetch_bytes(&client, &link, Some(u.as_str())).await
                                {
                                    if bytes.len() > 12 * 1024 * 1024 {
                                        continue; // skip very large PDFs
                                    }
                                    if let Ok(txt) = pdf_bytes_to_text(&bytes) {
                                        if !txt.trim().is_empty() {
                                            out.push((link_key.clone(), txt));
                                            bar.inc(1);
                                            added += 1;
                                        }
                                    }
                                }
                            }
                        } else if link_key.starts_with(scope_prefix) {
                            q.push_back((link, d + 1, Some(u.as_str().to_string())));
                            added += 1;
                        }
                    }
                }
            }
            Err(_) => { /* ignore fetch errors */ }
        }
        // politeness delay
        sleep(Duration::from_millis(crawl_delay_ms)).await;
    }

    bar.finish_and_clear();
    Ok(out)
}

/// ================= Ollama API =================
#[derive(Serialize)]
struct EmbeddingsReq<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
}
#[derive(Deserialize)]
struct EmbeddingsResp {
    embedding: Vec<f32>,
}

async fn embed_text(ollama: &str, model: &str, text: &str) -> Anyhow<Vec<f32>> {
    if std::env::var("DISABLE_EMBEDDINGS").ok().as_deref() == Some("1") {
        return Ok(Vec::new());
    }

    let mut safe = clamp_for_embedding(text);
    let mut num_ctx: usize = std::env::var("EMBED_NUM_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);

    let client = reqwest::Client::new();
    let url = format!("{}/api/embeddings", ollama);

    let mut tries = 0usize;
    loop {
        let resp = client
            .post(&url)
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
            bail!(
                "Embeddings failed ({}): {}. Hint: adjust CHUNK_TARGET_CHARS/EMBED_MAX_CHARS or pick a bigger-context embedding model.",
                status,
                body
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
struct GenerateChunk {
    response: Option<String>,
}

async fn generate(ollama: &str, model: &str, prompt: &str, temperature: f32) -> Anyhow<String> {
    let mut res = reqwest::Client::new()
        .post(format!("{}/api/generate", ollama))
        .json(&GenerateReq {
            model,
            prompt,
            temperature: Some(temperature.clamp(0.0, 1.0)),
            stream: true,
        })
        .send()
        .await?
        .error_for_status()?;

    let mut out = String::new();
    while let Some(chunk) = res.chunk().await? {
        let line = String::from_utf8_lossy(&chunk).to_string();
        for part in line.lines() {
            if part.trim().is_empty() {
                continue;
            }
            if let Ok(tick) = serde_json::from_str::<GenerateChunk>(part) {
                if let Some(s) = tick.response {
                    out.push_str(&s);
                }
            }
        }
    }
    Ok(out)
}

/// ================= Lexical & BM25 =================
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
    if !cur.is_empty() {
        out.push(cur);
    }
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
    if ql.contains("ects")
        || ql.contains("credit")
        || ql.contains("credits")
        || ql.contains("points")
    {
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
    if total_docs == 0 || avg_len == 0.0 {
        return 0.0;
    }
    let k1 = 1.5_f32;
    let b = 0.75_f32;

    let mut score = 0.0_f32;
    for term in q_terms {
        let f = *chunk.tf.get(term).unwrap_or(&0) as f32;
        if f <= 0.0 {
            continue;
        }
        let df_t = *df.get(term).unwrap_or(&0) as f32;
        if df_t <= 0.0 {
            continue;
        }
        let idf = ((total_docs as f32 - df_t + 0.5) / (df_t + 0.5) + 1e-6).ln();
        let denom = f + k1 * (1.0 - b + b * (chunk.tok_len as f32 / avg_len));
        score += idf * (f * (k1 + 1.0) / denom);
    }
    score
}

/// ================= Index build/extend & Hybrid RAG =================
fn sip_hash_u64(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

async fn chunks_from_pairs(
    ollama: &str,
    embed_model: &str,
    pairs: Vec<(String, String)>,
) -> Anyhow<(Vec<Chunk>, HashMap<String, u32>, usize, usize)> {
    let mut chunks = Vec::new();
    let mut df: HashMap<String, u32> = HashMap::new();
    let mut total_len: usize = 0;
    let mut total_docs: usize = 0;

    let mut seen_texts: HashSet<u64> = HashSet::new();
    let target = embed_chunk_size(); // default ~600

    for (url, text) in pairs {
        for (i, piece) in chunk_text(&text, target, 120).into_iter().enumerate() {
            // de-dup identical pieces in-session to avoid re-embedding
            let h = sip_hash_u64(&piece);
            if !seen_texts.insert(h) {
                continue;
            }

            let tokens = tokenize_lower(&piece);
            let tf = bow_tf(&tokens);
            let tok_len = tokens.len();
            total_len += tok_len;

            // DF update
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
            total_docs += 1;
        }
    }
    Ok((chunks, df, total_len, total_docs))
}

async fn build_index(
    ollama: &str,
    embed_model: &str,
    gen_model: &str,
    pairs: Vec<(String, String)>,
    scope: String,
) -> Anyhow<IndexFile> {
    let (chunks, df, total_len, total_docs) = chunks_from_pairs(ollama, embed_model, pairs).await?;
    let avg_len = if total_docs == 0 {
        0.0
    } else {
        total_len as f32 / total_docs as f32
    };

    Ok(IndexFile {
        embed_model: embed_model.to_string(),
        gen_model: gen_model.to_string(),
        chunks,
        created_at: Utc::now().to_rfc3339(),
        source_scope: scope,
        df,
        total_docs,
        avg_len,
    })
}

fn extend_index(
    idx: &mut IndexFile,
    new_chunks: Vec<Chunk>,
    new_df: HashMap<String, u32>,
    new_total_len: usize,
    new_docs: usize,
) {
    for (term, add) in new_df {
        *idx.df.entry(term).or_insert(0) += add;
    }
    let prev_docs = idx.total_docs;
    idx.total_docs += new_docs;
    let total_len_prev = (idx.avg_len * prev_docs as f32) as usize;
    let total_len_new = total_len_prev + new_total_len;
    idx.avg_len = if idx.total_docs == 0 {
        0.0
    } else {
        total_len_new as f32 / idx.total_docs as f32
    };
    idx.chunks.extend(new_chunks);
}

/// hybrid rerank
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

    let mut prelim: Vec<(&Chunk, f32)> =
        chunks.iter().map(|c| (c, cosine(emb_q, &c.embedding))).collect();
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

fn keyword_bonus(text: &str, url: &str, q: &str) -> f32 {
    let t = text.to_ascii_lowercase();
    let u = url.to_ascii_lowercase();
    let ql = q.to_ascii_lowercase();
    let mut s: f32 = 0.0;

    for h in [
        "ects",
        "credit",
        "credits",
        "thesis",
        "module",
        "modules",
        "study plan",
        "curriculum",
        "program structure",
        "pflichtbereich",
        "wahlpflichtbereich",
    ] {
        if ql.contains(h) && (t.contains(h) || u.contains(h)) {
            s += 0.9;
        }
    }
    for h in [
        "uni-assist",
        "uni assist",
        "aps",
        "application deadline",
        "admissions office",
        "studierendensekretariat",
    ] {
        if ql.contains(h) && (t.contains(h) || u.contains(h)) {
            s += 0.8;
        }
    }
    if (ql.contains("english") || ql.contains("program"))
        && (t.contains("master of science")
            || t.contains("master of arts")
            || t.contains("master of laws")
            || t.contains("english-taught"))
    {
        s += 0.6;
    }
    if ql.contains("who")
        || ql.contains("incharge")
        || ql.contains("in charge")
        || ql.contains("responsible")
        || ql.contains("contact")
    {
        if Regex::new(r"[A-Z][a-z]+ [A-Z][a-z]+").unwrap().is_match(&t) {
            s += 0.5;
        }
        if t.contains('@') || Regex::new(r"room\s*\d+").unwrap().is_match(&t) {
            s += 0.3;
        }
    }
    let num_re = Regex::new(r"\b\d{1,3}\b").unwrap();
    for m in num_re.find_iter(&ql) {
        let num = m.as_str();
        if t.contains(num) || u.contains(num) {
            s += 0.2;
        }
    }
    s.min(2.0)
}

fn choose_primary_source(picks: &[(&Chunk, f32)]) -> String {
    if let Some((chunk, _)) = picks.first() {
        chunk.url.clone()
    } else {
        String::new()
    }
}

/// prompt (comprehensive answer)
fn build_prompt(question: &str, contexts: &[(&Chunk, f32)], primary_source: &str) -> String {
    let mut ctx = String::new();
    for (c, _) in contexts {
        ctx.push_str(&format!("SOURCE URL: {}\n{}\n\n", c.url, c.text));
    }

    let ql = question.to_ascii_lowercase();
    let wants_list = ql.contains("list") || ql.contains("which program");
    let wants_deadline = ql.contains("deadline") || ql.contains("closing date");
    let wants_contact = ql.contains("who") || ql.contains("contact") || ql.contains("in charge");
    let wants_require = ql.contains("requirement")
        || ql.contains("eligibility")
        || ql.contains("admission")
        || ql.contains("uni-assist")
        || ql.contains("aps");

    let mut rules = String::new();
    if wants_list {
        rules.push_str("- If the question asks for a list, provide a complete bullet list using the exact titles/names found in CONTEXT.\n");
    }
    if wants_deadline {
        rules.push_str(
            "- Give exact dates first (with semester labels if present), and specify the portal (e.g., uni-assist vs. university) if stated.\n",
        );
    }
    if wants_contact {
        rules.push_str(
            "- Include full contact details if present: name, role, office/room, email/phone.\n",
        );
    }
    if wants_require {
        rules.push_str(
            "- If requirements are present, include a clear checklist (degree, language level, uni-assist/APS, documents).\n",
        );
    }

    let global_rules = r#"- Use ONLY the CONTEXT. Do NOT invent details.
- Quote exact numbers, dates, names and program titles.
- Prefer concise paragraphs and bullet points. Use short headings if helpful.
- Include short quotes only when needed to preserve exact wording.
- End with one source line:  Source: <URL>."#;

    let primary = if primary_source.is_empty() {
        "(unknown)"
    } else {
        primary_source
    };
    let rules_view: &str = if rules.is_empty() { "(none)" } else { &rules };

    format!(
        r#"You are an expert assistant.

{global}

Additional guidance:
{rules}

QUESTION:
{q}

CONTEXT:
{ctx}

Write a comprehensive, precise answer strictly from the CONTEXT. Be complete (not a 3-point summary). Use clear paragraphs and bullets where helpful. End with:
Source: {primary}
"#,
        q = question,
        ctx = ctx,
        global = global_rules,
        rules = rules_view,
        primary = primary
    )
}

/// ================= HTTP types =================
#[derive(Deserialize)]
struct IndexManyReq {
    session_id: String,
    urls: Vec<String>,
    depth: Option<usize>,
    max_pages: Option<usize>,
    scope_prefix: Option<String>,
}
#[derive(Serialize)]
struct IndexResp {
    ok: bool,
    chunks: usize,
    pages_indexed: usize,
    created_at: String,
    source_scope: String,
}

#[derive(Deserialize)]
struct AskReq {
    session_id: String,
    question: String,
    top_k: Option<usize>,
    temperature: Option<f32>,
}
#[derive(Serialize)]
struct AskResp {
    answer: String,
    sources: Vec<String>,
}

/// ================= Handlers =================
async fn index_many(State(st): State<AppState>, Json(req): Json<IndexManyReq>) -> impl IntoResponse {
    if req.urls.is_empty() {
        return (StatusCode::BAD_REQUEST, "Provide at least one URL").into_response();
    }
    // Sanitize first
    let mut starts: Vec<Url> = Vec::new();
    for u in &req.urls {
        match sanitize_url(u) {
            Ok(url) => starts.push(url),
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("Invalid URL `{u}`: {e}"))
                    .into_response()
            }
        }
    }

    let depth = req.depth.filter(|d| *d > 0).unwrap_or(3);
    let max_pages = req.max_pages.filter(|m| *m > 0).unwrap_or(200);

    // Default scope: host of FIRST URL
    let scope = req
        .scope_prefix
        .unwrap_or_else(|| starts[0][..Position::BeforePath].to_string());

    // Crawl each start and gather (url,text)
    let mut all_pairs: Vec<(String, String)> = Vec::new();
    for start in &starts {
        let pairs = match crawl(start, depth, &scope, max_pages).await {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Crawl failed for {}: {e:#}", start),
                )
                    .into_response()
            }
        };
        all_pairs.extend(pairs);
    }

    if all_pairs.is_empty() {
        return (StatusCode::BAD_REQUEST, "Crawl returned 0 pages").into_response();
    }

    // If session exists -> extend, else build
    let mut sessions = st.sessions.write().await;
    if let Some(idx) = sessions.get_mut(&req.session_id) {
        match chunks_from_pairs(&st.ollama_host, &st.embed_model, all_pairs).await {
            Ok((new_chunks, new_df, new_total_len, new_docs)) => {
                extend_index(idx, new_chunks, new_df, new_total_len, new_docs);
                let pages = idx
                    .chunks
                    .iter()
                    .map(|c| &c.url)
                    .collect::<HashSet<_>>()
                    .len();
                let resp = IndexResp {
                    ok: true,
                    chunks: idx.chunks.len(),
                    pages_indexed: pages,
                    created_at: idx.created_at.clone(),
                    source_scope: idx.source_scope.clone(),
                };
                return Json(resp).into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Index extend failed: {e:#}"),
                )
                    .into_response()
            }
        }
    } else {
        let scope_copy = scope.clone();
        match build_index(
            &st.ollama_host,
            &st.embed_model,
            &st.gen_model,
            all_pairs,
            scope_copy,
        )
        .await
        {
            Ok(idx) => {
                let pages = idx
                    .chunks
                    .iter()
                    .map(|c| &c.url)
                    .collect::<HashSet<_>>()
                    .len();
                let resp = IndexResp {
                    ok: true,
                    chunks: idx.chunks.len(),
                    pages_indexed: pages,
                    created_at: idx.created_at.clone(),
                    source_scope: idx.source_scope.clone(),
                };
                sessions.insert(req.session_id, idx);
                Json(resp).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Index failed: {e:#}"),
            )
                .into_response(),
        }
    }
}

async fn upload_files(State(st): State<AppState>, mut mp: Multipart) -> impl IntoResponse {
    // Expect: session_id + one or more files
    let mut session_id: Option<String> = None;
    let mut files_saved: Vec<PathBuf> = Vec::new();

    // Important: single staging dir lives for whole handler
    let staging = match tempdir() {
        Ok(d) => d,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Temp dir error: {e}")).into_response(),
    };

    while let Ok(Some(field)) = mp.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "session_id" {
            let v = field.text().await.unwrap_or_default();
            if !v.trim().is_empty() {
                session_id = Some(v.trim().to_string());
            }
            continue;
        }
        if name == "files" {
            let fname = field
                .file_name()
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("upload-{}.bin", uuid_like()));
            let bytes = match field.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read upload `{fname}`: {e}"),
                    )
                        .into_response()
                }
            };
            let path = staging.path().join(&fname);
            match fs::File::create(&path).and_then(|mut f| f.write_all(&bytes).map(|_| f)) {
                Ok(_) => files_saved.push(path),
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to save `{fname}`: {e}"),
                    )
                        .into_response()
                }
            }
        }
    }

    let session_id = match session_id {
        Some(s) => s,
        None => return (StatusCode::BAD_REQUEST, "Missing session_id").into_response(),
    };

    if files_saved.is_empty() {
        return (StatusCode::BAD_REQUEST, "No files uploaded").into_response();
    }

    // Extract -> (logical-url, text) while staging is alive
    let mut pairs: Vec<(String, String)> = Vec::new();
    for p in &files_saved {
        match extract_any_file_to_text(p) {
            Ok(txt) => {
                if !txt.trim().is_empty() {
                    let logical = format!("file://{}", p.display());
                    pairs.push((logical, txt));
                }
            }
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to extract `{}`: {e}", p.display()),
                )
                    .into_response()
            }
        }
    }

    if pairs.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            "No text extracted from uploads",
        )
            .into_response();
    }

    // Insert/extend index for session
    let mut sessions = st.sessions.write().await;
    if let Some(idx) = sessions.get_mut(&session_id) {
        match chunks_from_pairs(&st.ollama_host, &st.embed_model, pairs).await {
            Ok((new_chunks, new_df, new_total_len, new_docs)) => {
                extend_index(idx, new_chunks, new_df, new_total_len, new_docs);
                let pages = idx
                    .chunks
                    .iter()
                    .map(|c| &c.url)
                    .collect::<HashSet<_>>()
                    .len();
                let resp = serde_json::json!({
                    "ok": true,
                    "files_processed": files_saved.len(),
                    "chunks": idx.chunks.len(),
                    "pages_indexed": pages
                });
                // staging drops here, after extraction 👍
                return (StatusCode::OK, axum::Json(resp)).into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Index extend failed: {e:#}"),
                )
                    .into_response()
            }
        }
    } else {
        match build_index(
            &st.ollama_host,
            &st.embed_model,
            &st.gen_model,
            pairs,
            "(uploads)".to_string(),
        )
        .await
        {
            Ok(idx) => {
                let pages = idx
                    .chunks
                    .iter()
                    .map(|c| &c.url)
                    .collect::<HashSet<_>>()
                    .len();
                let resp = serde_json::json!({
                    "ok": true,
                    "files_processed": files_saved.len(),
                    "chunks": idx.chunks.len(),
                    "pages_indexed": pages
                });
                sessions.insert(session_id, idx);
                // staging drops here, after insertion 👍
                return (StatusCode::OK, axum::Json(resp)).into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Index failed: {e:#}"),
                )
                    .into_response()
            }
        }
    }
}

fn uuid_like() -> String {
    use rand::RngCore;
    let mut b = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut b);
    hex::encode(b)
}

fn extract_any_file_to_text(path: &PathBuf) -> Anyhow<String> {
    let lower = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match lower.as_str() {
        "pdf" => {
            let bytes = fs::read(path)?;
            pdf_bytes_to_text(&bytes)
        }
        "txt" | "md" | "html" | "htm" => {
            let s = fs::read_to_string(path)?;
            if lower == "html" || lower == "htm" {
                let base = Url::parse("https://local.file/").unwrap();
                let (t, _) = extract_text_and_links(&base, &s);
                Ok(t)
            } else {
                Ok(normalize_ws(&s))
            }
        }
        "docx" | "pptx" | "odt" => {
            if !have_cmd("pandoc") {
                bail!(
                    "pandoc not found; install pandoc to extract {}",
                    path.display()
                );
            }
            let out = Command::new("pandoc").arg(path).arg("-t").arg("plain").output()?;
            if !out.status.success() {
                bail!("pandoc failed on {}", path.display());
            }
            Ok(normalize_ws(&String::from_utf8_lossy(&out.stdout)))
        }
        _ => {
            if have_cmd("pandoc") {
                let out = Command::new("pandoc").arg(path).arg("-t").arg("plain").output()?;
                if !out.status.success() {
                    bail!("pandoc failed on {}", path.display());
                }
                Ok(normalize_ws(&String::from_utf8_lossy(&out.stdout)))
            } else {
                bail!("Unsupported file type `{}` and pandoc not installed", lower);
            }
        }
    }
}

async fn ask(State(st): State<AppState>, Json(req): Json<AskReq>) -> impl IntoResponse {
    let idx = {
        let sessions = st.sessions.read().await;
        match sessions.get(&req.session_id) {
            Some(i) => i.clone(),
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    "No index for this session. Call /api/index_many and/or /api/upload first.",
                )
                    .into_response()
            }
        }
    };

    let emb_q = match embed_text(&st.ollama_host, &idx.embed_model, &req.question).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Embed failed: {e:#}"),
            )
                .into_response()
        }
    };

    let ql = req.question.to_ascii_lowercase();
    let list_programs = (ql.contains("english") || ql.contains("in english"))
        && (ql.contains("program") || ql.contains("study program") || ql.contains("list"));

    let default_k = if list_programs { 30 } else { 18 };
    let retrieval_k = req.top_k.unwrap_or(default_k);
    let picks = rerank_hybrid(
        &req.question,
        &emb_q,
        &idx.chunks,
        &idx.df,
        idx.total_docs,
        idx.avg_len,
        retrieval_k.min(default_k),
    );

    if picks.is_empty() {
        return Json(AskResp {
            answer: "I couldn’t retrieve any relevant context from the current index.".to_string(),
            sources: vec![],
        })
        .into_response();
    }

    let primary_link = choose_primary_source(&picks);
    let prompt = build_prompt(&req.question, &picks, &primary_link);
    let temperature = if list_programs {
        0.0
    } else {
        req.temperature.unwrap_or(0.25)
    };

    let mut answer = match generate(&st.ollama_host, &idx.gen_model, &prompt, temperature).await {
        Ok(a) => a,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Generation failed: {e:#}"),
            )
                .into_response()
        }
    };

    let mut seen = HashSet::new();
    let mut sources: Vec<String> = Vec::new();
    if !primary_link.is_empty() && seen.insert(primary_link.clone()) {
        sources.push(primary_link.clone());
    }
    for (c, _) in &picks {
        if seen.insert(c.url.clone()) {
            sources.push(c.url.clone());
        }
        if sources.len() >= 8 {
            break;
        }
    }

    if !answer.to_ascii_lowercase().contains("source:") {
        if let Some(first) = sources.first() {
            answer.push_str("\n\nSource: ");
            answer.push_str(first);
        }
    }

    Json(AskResp { answer, sources }).into_response()
}

/// ================= Static HTML =================
async fn index_html() -> impl IntoResponse {
    Html(include_str!("../static/index.html"))
}

/// ================= Main =================
#[tokio::main]
async fn main() -> Anyhow<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    let state = AppState {
        ollama_host: cli.ollama_host,
        embed_model: cli.embed_model,
        gen_model: cli.gen_model,
        sessions: Arc::new(RwLock::new(HashMap::new())),
    };

    // Only raise the body limit on the upload route
    let app = Router::new()
        .route("/", get(index_html))
        .route("/api/index_many", post(index_many))
        .route(
            "/api/upload",
            post(upload_files).route_layer(DefaultBodyLimit::max(50 * 1024 * 1024)),
        )
        .route("/api/ask", post(ask))
        .with_state(state);

    let addr: SocketAddr = cli.bind.parse()?;
    println!("➡️  Open http://{addr}/");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

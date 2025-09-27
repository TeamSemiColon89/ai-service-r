use actix_web::{post, web, App, HttpResponse, HttpServer, Error};
use env_logger::Env;
use log::info;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokenizers::{Tokenizer, TruncationDirection, PaddingDirection};
use tokio::sync::Mutex;

use ort::execution_providers::CPUExecutionProvider;
use ort::session::Session;
use ort::value::Value;

const MAX_LEN: usize = 128;

struct AppState {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
}

#[derive(Deserialize)]
struct GenericPayload {
    msg: Option<String>,
    payload: Option<String>,
    data: Option<String>,
}

fn extract_text(body: &str) -> String {
    if let Ok(parsed) = serde_json::from_str::<GenericPayload>(body) {
        parsed.msg
            .or(parsed.payload)
            .or(parsed.data)
            .unwrap_or_else(|| body.to_string())
    } else {
        body.to_string()
    }
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

#[post("/decision{tail:.*}")]
async fn decision(body: String, data: web::Data<AppState>) -> Result<HttpResponse, Error> {
    info!("📥 Raw request body: {}", body);

    let text = extract_text(&body);
    info!("📥 Extracted text for analysis: {}", text);

    // Tokenize with truncation + padding
    let mut enc = data
        .tokenizer
        .encode(text.clone(), true)
        .map_err(|e| {
            actix_web::error::ErrorBadRequest(json!({
                "error": "tokenization_failed",
                "details": e.to_string()
            }))
        })?;

    enc.truncate(MAX_LEN, 0, TruncationDirection::Right);
    enc.pad(MAX_LEN, 0, 0, "[PAD]", PaddingDirection::Right);

    let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
    let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&v| v as i64).collect();

    info!("🔹 Final IDs len={}, sample={:?}", ids.len(), &ids[..10.min(ids.len())]);
    info!("🔹 Final Mask len={}, sample={:?}", mask.len(), &mask[..10.min(mask.len())]);

    let shape = vec![1_i64, MAX_LEN as i64];

    let input_ids = Value::from_array((shape.clone(), ids))
        .map_err(|e| actix_web::error::ErrorInternalServerError(json!({
            "error": "failed_build_input_ids",
            "details": e.to_string()
        })))?;

    let attention_mask = Value::from_array((shape, mask))
        .map_err(|e| actix_web::error::ErrorInternalServerError(json!({
            "error": "failed_build_attention_mask",
            "details": e.to_string()
        })))?;

    // Run inference
    let logits: Vec<f32> = {
        let mut session = data.session.lock().await;
        let outputs = session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        })
        .map_err(|e| actix_web::error::ErrorInternalServerError(json!({
            "error": "inference_failed",
            "details": e.to_string()
        })))?;

        info!("🔹 ONNX output keys = {:?}", outputs.keys().collect::<Vec<_>>());

        let value = outputs.get("logits").ok_or_else(|| {
            actix_web::error::ErrorInternalServerError(json!({
                "error": "missing_output",
                "details": "expected 'logits' output not found"
            }))
        })?;

        let (_, slice) = value.try_extract_tensor::<f32>()
            .map_err(|e| actix_web::error::ErrorInternalServerError(json!({
                "error": "extract_logits_failed",
                "details": e.to_string()
            })))?;

        slice.to_vec()
    };

    if logits.len() < 2 {
        return Err(actix_web::error::ErrorInternalServerError(json!({
            "error": "unexpected_logits_shape",
            "details": format!("len={}", logits.len())
        })));
    }

    let logits_pair = [logits[0], logits[1]];
    let probs = softmax(&logits_pair);
    let non_sqli_prob = probs[0];
    let sqli_prob = probs[1];

    info!(
        "✅ Decision result: logits={:?}, probs={:?}, text={}",
        logits_pair, probs, text
    );

    if sqli_prob > non_sqli_prob {
        Ok(HttpResponse::Forbidden()
            .append_header(("x-ai-reason",
                format!("sqli_prob={:.4},non_sqli_prob={:.4}", sqli_prob, non_sqli_prob)))
            .json(json!({
                "allow": false,
                "reason": "sqli_detected",
                "logits": logits_pair,
                "probs": probs
            })))
    } else {
        Ok(HttpResponse::Ok()
            .append_header(("x-ai-reason",
                format!("sqli_prob={:.4},non_sqli_prob={:.4}", sqli_prob, non_sqli_prob)))
            .json(json!({
                "allow": true,
                "reason": "ok",
                "logits": logits_pair,
                "probs": probs
            })))
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let providers = [CPUExecutionProvider::default().into()];
    let session = Session::builder()
        .expect("SessionBuilder")
        .with_execution_providers(&providers)
        .expect("execution providers")
        .commit_from_file("models/distilbert_sqli.onnx")
        .expect("load model");

    let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json")
        .expect("load tokenizer");

    let state = web::Data::new(AppState {
        session: Arc::new(Mutex::new(session)),
        tokenizer: Arc::new(tokenizer),
    });

    println!("🚀 AI service running on http://0.0.0.0:5000/decision");

    HttpServer::new(move || App::new().app_data(state.clone()).service(decision))
        .bind(("0.0.0.0", 5000))?
        .run()
        .await
}

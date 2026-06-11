mod audio;
mod asr;
mod config;
mod nlu;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::Parser;
use tokio::net::TcpListener;

#[derive(Parser)]
#[command(name = "voice-infer", about = "ASR + NLU ONNX inference server")]
struct Args {
    /// ASR 模型目录
    #[arg(long, default_value = "models/asr")]
    asr_model_dir: PathBuf,

    /// NLU 模型目录
    #[arg(long, default_value = "models/nlu")]
    nlu_model_dir: PathBuf,

    /// NLU tokenizer 目录 (默认 {nlu_model_dir}/tokenizer)
    #[arg(long)]
    nlu_tokenizer_dir: Option<PathBuf>,

    /// 监听地址
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// ASR 服务端口
    #[arg(long, default_value_t = 8000)]
    asr_port: u16,

    /// NLU 服务端口
    #[arg(long, default_value_t = 8001)]
    nlu_port: u16,

    /// 使用 GPU (CUDA)
    #[arg(long)]
    gpu: bool,

    /// ONNX 推理线程数
    #[arg(long)]
    threads: Option<usize>,

    /// 只启动 ASR
    #[arg(long)]
    asr_only: bool,

    /// 只启动 NLU
    #[arg(long)]
    nlu_only: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    // 初始化 ONNX Runtime
    validate_ort_dylib_path()?;
    tracing::info!("initializing ONNX Runtime");
    let _ = ort::init().commit();
    tracing::info!("ONNX Runtime initialized");

    let nlu_tokenizer_dir = args.nlu_tokenizer_dir
        .unwrap_or_else(|| args.nlu_model_dir.join("tokenizer"));

    // 加载引擎
    let asr_engine = if !args.nlu_only {
        tracing::info!("loading ASR engine...");
        Some(Arc::new(Mutex::new(
            asr::AsrEngine::new(&args.asr_model_dir, args.threads, args.gpu)
                .context("loading ASR engine")?
        )))
    } else {
        None
    };

    let nlu_engine = if !args.asr_only {
        tracing::info!("loading NLU engine...");
        Some(Arc::new(Mutex::new(
            nlu::NluEngine::new(&args.nlu_model_dir, &nlu_tokenizer_dir, args.threads, args.gpu)
                .context("loading NLU engine")?
        )))
    } else {
        None
    };

    // 启动 HTTP 服务
    let mut handles = Vec::new();

    if let Some(engine) = asr_engine {
        let addr: SocketAddr = format!("{}:{}", args.host, args.asr_port).parse()?;
        let listener = TcpListener::bind(addr).await
            .with_context(|| format!("binding ASR to {addr}"))?;
        tracing::info!("ASR service: http://{addr}/asr");
        let router = asr::server::router(engine);
        handles.push(tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .context("ASR server")
        }));
    }

    if let Some(engine) = nlu_engine {
        let addr: SocketAddr = format!("{}:{}", args.host, args.nlu_port).parse()?;
        let listener = TcpListener::bind(addr).await
            .with_context(|| format!("binding NLU to {addr}"))?;
        tracing::info!("NLU service: http://{addr}/nlu");
        let router = nlu::server::router(engine);
        handles.push(tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .context("NLU server")
        }));
    }

    if handles.is_empty() {
        anyhow::bail!("no service to start (--asr-only and --nlu-only are mutually exclusive)");
    }

    // 等待任一服务结束
    let (result, _, _) = futures::future::select_all(handles).await;
    result??;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("shutting down");
}

fn validate_ort_dylib_path() -> Result<()> {
    match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) if Path::new(&path).is_file() => Ok(()),
        Ok(path) => anyhow::bail!(
            "ORT_DYLIB_PATH points to a missing file: {path}. Set it to the target ONNX Runtime dynamic library."
        ),
        Err(_) => {
            tracing::warn!(
                "ORT_DYLIB_PATH is not set; ort load-dynamic will use its default loader path"
            );
            Ok(())
        }
    }
}

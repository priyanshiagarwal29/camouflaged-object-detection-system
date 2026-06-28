import { useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const fileMeta = useMemo(() => {
    if (!file) return null;

    return {
      name: file.name,
      size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
      type: file.type || "image",
    };
  }, [file]);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
      if (result) URL.revokeObjectURL(result);
    };
  }, [preview, result]);

  const pickFile = (selectedFile) => {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please choose a valid image file.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);
    if (result) URL.revokeObjectURL(result);

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult("");
    setError("");
  };

  const handleFileChange = (event) => {
    pickFile(event.target.files?.[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    pickFile(event.dataTransfer.files?.[0]);
  };

  const handleDetect = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Detection failed with status ${response.status}`);
      }

      const blob = await response.blob();
      if (result) URL.revokeObjectURL(result);
      setResult(URL.createObjectURL(blob));
    } catch (requestError) {
      setError(
        requestError.message ||
          "Could not connect to the SINet-V2 backend. Make sure it is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetWorkspace = () => {
    if (preview) URL.revokeObjectURL(preview);
    if (result) URL.revokeObjectURL(result);

    setFile(null);
    setPreview("");
    setResult("");
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  return (
    <main className="app-shell">
      <section className="hero-section" aria-labelledby="page-title">
        <div className="hero-copy">
          <span className="eyebrow">SINet-V2 vision workspace</span>
          <h1 id="page-title">Camouflaged Object Detection</h1>
          <p>
            Upload a natural scene and send it to your backend model to reveal
            hidden objects with a clean side-by-side result view.
          </p>

          <div className="hero-actions">
            <button className="primary-button" onClick={() => inputRef.current?.click()}>
              Choose image
            </button>
            <button className="ghost-button" onClick={resetWorkspace} disabled={!file}>
              Clear
            </button>
          </div>
        </div>

        <div className="signal-panel" aria-hidden="true">
          <div className="radar-frame">
            <div className="scan-line" />
            <div className="target target-one" />
            <div className="target target-two" />
            <div className="target target-three" />
          </div>
          <div className="signal-readout">
            <span>mask confidence</span>
            <strong>{loading ? "scanning" : result ? "ready" : "standby"}</strong>
          </div>
        </div>
      </section>

      <section className="workspace" aria-label="Detection workspace">
        <div className="control-column">
          <div
            className={`upload-zone ${isDragging ? "is-dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <input
              ref={inputRef}
              className="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />

            <div className="upload-icon">+</div>
            <h2>Drop an image here</h2>
            <p>PNG, JPG, JPEG, or WEBP scene images work best.</p>
            <button className="secondary-button" onClick={() => inputRef.current?.click()}>
              Browse files
            </button>
          </div>

          {fileMeta && (
            <div className="file-card">
              <div>
                <span>Selected image</span>
                <strong>{fileMeta.name}</strong>
              </div>
              <small>
                {fileMeta.size} · {fileMeta.type}
              </small>
            </div>
          )}

          {error && <div className="error-banner">{error}</div>}

          <button
            className="detect-button"
            onClick={handleDetect}
            disabled={!file || loading}
          >
            {loading ? "Analyzing image..." : "Run detection"}
          </button>
        </div>

        <div className="result-grid">
          <PreviewCard
            label="Original"
            image={preview}
            emptyText="Your source image will appear here."
          />
          <PreviewCard
            label="SINet-V2 result"
            image={result}
            emptyText={loading ? "The model is generating the mask." : "Run detection to see the result."}
            isLoading={loading}
            downloadName={file ? `${file.name.replace(/\.[^.]+$/, "")}-sinet-result.png` : ""}
          />
        </div>
      </section>
    </main>
  );
}

function PreviewCard({ label, image, emptyText, isLoading = false, downloadName = "" }) {
  return (
    <article className="preview-card">
      <div className="card-header">
        <h2>{label}</h2>
        {image && downloadName && (
          <a className="download-link" href={image} download={downloadName}>
            Download
          </a>
        )}
      </div>

      <div className="image-frame">
        {image ? (
          <img src={image} alt={`${label} preview`} />
        ) : (
          <div className={`empty-state ${isLoading ? "is-loading" : ""}`}>
            <span />
            <p>{emptyText}</p>
          </div>
        )}
      </div>
    </article>
  );
}

export default App;

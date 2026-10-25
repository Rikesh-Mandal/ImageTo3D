"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useState, type ChangeEvent, type FormEvent } from "react";

import MeshViewer from "@/components/MeshViewer";

type ReconstructionResult = {
  id: string;
  status: string;
  device: string;
  vertices: number;
  triangles: number;
  triangles_rejected: number;
  depth_map_url: string;
  mesh_url: string;
};

type ResultTab = "mesh" | "depth";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [includeBackground, setIncludeBackground] = useState(false);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState<string | null>(null);

  const [result, setResult] = useState<ReconstructionResult | null>(null);

  const [activeTab, setActiveTab] = useState<ResultTab>("mesh");

  // Clean up temporary browser image URLs.
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;

    setError(null);
    setResult(null);

    if (!selected) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    if (selected.type !== "image/jpeg" && selected.type !== "image/png") {
      setError("Please select a JPG or PNG image.");
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    if (selected.size > MAX_FILE_SIZE) {
      setError("Image size cannot exceed 10 MB.");
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  async function handleGenerate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();

      formData.append("file", file);

      formData.append("include_background", String(includeBackground));

      const response = await fetch("/api/reconstructions", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Reconstruction failed: HTTP ${response.status}`,
        );
      }

      if (
        typeof data.id !== "string" ||
        typeof data.depth_map_url !== "string" ||
        typeof data.mesh_url !== "string"
      ) {
        throw new Error("The backend returned an invalid result.");
      }

      setResult(data as ReconstructionResult);
      setActiveTab("mesh");
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "An unexpected error occurred.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main
      className="
        flex min-h-screen flex-col bg-[#080d18]
        px-4 py-3 text-slate-100
        lg:h-dvh lg:overflow-hidden
        lg:px-6
      "
    >
      <div
        className="
          mx-auto flex w-full max-w-[1600px]
          flex-1 flex-col
          lg:min-h-0
        "
      >
        {/* COMPACT HEADER */}
        <header
          className="
            mb-3 flex shrink-0 flex-wrap
            items-center justify-between gap-3
            border-b border-white/10 pb-3
          "
        >
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Monocular 3D Reconstruction
            </p>

            <h1 className="text-2xl font-bold tracking-tight">ImageTo3D</h1>

            <p className="text-xs text-slate-400">
              Generate a depth map and visible-surface 3D mesh from a single
              photograph.
            </p>
          </div>

          <div
            className="
              rounded-full border border-cyan-500/20
              bg-cyan-500/10 px-3 py-1.5
              text-xs font-medium text-cyan-300
            "
          >
            ResNet-50 · Open3D · CPU
          </div>
        </header>

        {/* MAIN TWO-COLUMN AREA */}
        <div
          className="
            grid grid-cols-1 gap-4
            lg:min-h-0 lg:flex-1
            lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]
          "
        >
          {/* LEFT PANEL: INPUT IMAGE */}
          <section
            className="
              flex min-w-0 flex-col overflow-hidden
              rounded-2xl border border-white/10
              bg-[#111a2a]
              lg:min-h-0
            "
          >
            {/* PANEL HEADER */}
            <div className="shrink-0 border-b border-white/10 px-4 py-3">
              <h2 className="text-base font-semibold">Input Image</h2>

              <p className="text-xs text-slate-400">
                Choose an RGB photograph.
              </p>
            </div>

            {/* RESPONSIVE IMAGE PREVIEW */}
            <div
              className="
                h-72 min-w-0 overflow-hidden
                bg-[#0b1220] p-3
                sm:h-80
                lg:h-auto lg:min-h-0 lg:flex-1
              "
            >
              <div
                className="
                  image-preview-frame
                  flex h-full w-full items-center
                  justify-center rounded-xl
                  border border-white/10
                  bg-[#080d18]
                "
              >
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Uploaded RGB photograph"
                    className="max-h-full max-w-full object-contain"
                  />
                ) : (
                  <div className="p-4 text-center">
                    <div className="mb-2 text-4xl text-slate-600">◇</div>

                    <p className="text-sm font-medium text-slate-300">
                      No image selected
                    </p>

                    <p className="mt-1 text-xs text-slate-500">
                      JPG or PNG · Maximum 10 MB
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* CONTROLS: ALWAYS VISIBLE ON DESKTOP */}
            <form
              onSubmit={handleGenerate}
              className="
                shrink-0 space-y-3
                border-t border-white/10 p-3
              "
            >
              <div>
                <label
                  htmlFor="input-image"
                  className="mb-1 block text-xs font-medium text-slate-300"
                >
                  Choose an image
                </label>

                <input
                  id="input-image"
                  type="file"
                  accept="image/jpeg,image/png"
                  disabled={loading}
                  onChange={handleFileChange}
                  className="
                    block w-full min-w-0 cursor-pointer
                    rounded-xl border border-dashed
                    border-slate-600 bg-[#0b1220]
                    p-2 text-xs text-slate-300
                    file:mr-3 file:cursor-pointer
                    file:rounded-lg file:border-0
                    file:bg-cyan-900 file:px-3 file:py-2
                    file:font-semibold file:text-cyan-200
                    disabled:opacity-50
                  "
                />

                {file && (
                  <p className="mt-1 truncate text-xs text-slate-400">
                    {file.name}
                  </p>
                )}
              </div>

              {/* CHECKBOX AND GENERATE BUTTON */}
              <div className="grid grid-cols-1 items-center gap-3 sm:grid-cols-[minmax(0,1fr)_auto]">
                <label className="flex cursor-pointer items-center gap-2">
                  <input
                    type="checkbox"
                    checked={includeBackground}
                    disabled={loading}
                    onChange={(event) => {
                      setIncludeBackground(event.target.checked);

                      setResult(null);
                    }}
                    className="h-4 w-4 shrink-0 accent-cyan-500"
                  />

                  <span className="text-xs text-slate-300">
                    Include background
                  </span>
                </label>

                <button
                  type="submit"
                  disabled={!file || loading}
                  className="
                    rounded-xl bg-cyan-500
                    px-5 py-3 text-center
                    text-sm font-semibold text-slate-950
                    transition hover:bg-cyan-400
                    disabled:cursor-not-allowed
                    disabled:bg-slate-700
                    disabled:text-slate-400
                  "
                >
                  {loading ? "Generating..." : "Generate 3D Model"}
                </button>
              </div>

              {error && (
                <div
                  role="alert"
                  className="
                    max-h-16 overflow-y-auto
                    rounded-lg border border-red-500/30
                    bg-red-500/10 p-2
                    text-xs text-red-200
                  "
                >
                  {error}
                </div>
              )}
            </form>
          </section>

          {/* RIGHT PANEL: RECONSTRUCTION RESULT */}
          <section
            className="
              flex min-w-0 flex-col overflow-hidden
              rounded-2xl border border-white/10
              bg-[#111a2a]
              lg:min-h-0
            "
          >
            {/* PANEL HEADER */}
            <div
              className="
                flex shrink-0 items-center
                justify-between gap-3
                border-b border-white/10
                px-4 py-3
              "
            >
              <div>
                <h2 className="text-base font-semibold">
                  Reconstruction Result
                </h2>

                <p className="text-xs text-slate-400">
                  Explore the generated depth and geometry.
                </p>
              </div>

              {result && (
                <span
                  className="
                    rounded-full bg-emerald-500/10
                    px-3 py-1 text-xs
                    font-medium text-emerald-300
                  "
                >
                  Completed
                </span>
              )}
            </div>

            {/* RESULT TABS */}
            <div
              className="
                flex shrink-0 gap-2
                border-b border-white/10
                px-3 pt-1
              "
            >
              <button
                type="button"
                disabled={!result}
                onClick={() => setActiveTab("mesh")}
                className={`
                  rounded-t-lg px-4 py-2
                  text-xs font-medium transition
                  disabled:cursor-not-allowed
                  disabled:opacity-40
                  ${
                    activeTab === "mesh"
                      ? "border-b-2 border-cyan-400 text-cyan-300"
                      : "text-slate-400 hover:text-white"
                  }
                `}
              >
                3D Mesh
              </button>

              <button
                type="button"
                disabled={!result}
                onClick={() => setActiveTab("depth")}
                className={`
                  rounded-t-lg px-4 py-2
                  text-xs font-medium transition
                  disabled:cursor-not-allowed
                  disabled:opacity-40
                  ${
                    activeTab === "depth"
                      ? "border-b-2 border-cyan-400 text-cyan-300"
                      : "text-slate-400 hover:text-white"
                  }
                `}
              >
                Depth Map
              </button>
            </div>

            {/* FLEXIBLE RESULT VIEWER */}
            <div
              className="
                h-72 min-w-0 overflow-hidden
                bg-[#0b1220]
                sm:h-80
                lg:h-auto lg:min-h-0 lg:flex-1
              "
            >
              {loading ? (
                <div
                  className="
                    flex h-full flex-col
                    items-center justify-center
                    gap-3 p-4 text-center
                  "
                >
                  <div
                    className="
                      h-9 w-9 animate-spin
                      rounded-full border-4
                      border-cyan-500/20
                      border-t-cyan-400
                    "
                  />

                  <p className="text-sm font-medium">
                    Generating reconstruction...
                  </p>

                  <p className="text-xs text-slate-400">
                    Running depth inference and mesh generation.
                  </p>
                </div>
              ) : !result ? (
                <div
                  className="
                    flex h-full flex-col
                    items-center justify-center
                    p-4 text-center
                  "
                >
                  <div className="mb-2 text-4xl text-slate-600">◇</div>

                  <p className="text-sm font-medium text-slate-300">
                    No reconstruction yet
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Your 3D model and depth map will appear here.
                  </p>
                </div>
              ) : activeTab === "mesh" ? (
                <div className="mesh-viewer-frame h-full w-full">
                  <MeshViewer key={result.id} url={result.mesh_url} />
                </div>
              ) : (
                <div className="image-preview-frame flex h-full w-full items-center justify-center p-3">
                  <img
                    src={result.depth_map_url}
                    alt="Predicted depth map"
                    className="max-h-full max-w-full object-contain"
                  />
                </div>
              )}
            </div>

            {/* STATS AND DOWNLOAD CONTROLS */}
            <div
              className="
                shrink-0 border-t
                border-white/10 p-3
              "
            >
              {result ? (
                <div className="space-y-3">
                  {/* GEOMETRY STATISTICS */}
                  <div className="grid grid-cols-3 gap-2">
                    <div className="rounded-lg bg-white/5 px-3 py-2">
                      <p className="text-[10px] text-slate-400">Vertices</p>

                      <p className="text-sm font-semibold">
                        {result.vertices.toLocaleString()}
                      </p>
                    </div>

                    <div className="rounded-lg bg-white/5 px-3 py-2">
                      <p className="text-[10px] text-slate-400">Triangles</p>

                      <p className="text-sm font-semibold">
                        {result.triangles.toLocaleString()}
                      </p>
                    </div>

                    <div className="rounded-lg bg-white/5 px-3 py-2">
                      <p className="text-[10px] text-slate-400">Rejected</p>

                      <p className="text-sm font-semibold">
                        {result.triangles_rejected.toLocaleString()}
                      </p>
                    </div>
                  </div>

                  {/* DOWNLOAD BUTTONS */}
                  <div className="grid grid-cols-2 gap-2">
                    <a
                      href={result.depth_map_url}
                      className="
                        rounded-lg border border-cyan-500/40
                        bg-cyan-500/10 px-3 py-2.5
                        text-center text-xs font-semibold
                        text-cyan-300
                        transition hover:bg-cyan-500/20
                      "
                    >
                      Download Depth Map
                    </a>

                    <a
                      href={result.mesh_url}
                      className="
                        rounded-lg bg-cyan-500
                        px-3 py-2.5 text-center
                        text-xs font-semibold text-slate-950
                        transition hover:bg-cyan-400
                      "
                    >
                      Download 3D Mesh
                    </a>
                  </div>
                </div>
              ) : (
                <div className="py-4 text-center text-xs text-slate-500">
                  Drag to rotate · Scroll to zoom
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

"use client";

import { useEffect, useState } from "react";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

type MeshViewerProps = {
  url: string;
};

/**
 * Convert 64-bit floating-point attributes into
 * WebGL-compatible 32-bit floating-point attributes.
 *
 * Open3D PLY files can contain double-precision vertices.
 * Newer Three.js PLYLoader versions preserve that precision,
 * but WebGL cannot directly render Float64Array attributes.
 */
function convertGeometryToFloat32(geometry: THREE.BufferGeometry): void {
  for (const attributeName of Object.keys(geometry.attributes)) {
    const attribute = geometry.getAttribute(attributeName);

    if (
      attribute instanceof THREE.BufferAttribute &&
      attribute.array instanceof Float64Array
    ) {
      const converted = new THREE.Float32BufferAttribute(
        attribute.array,
        attribute.itemSize,
        attribute.normalized,
      );

      converted.name = attribute.name;

      geometry.setAttribute(attributeName, converted);
    }
  }

  // Face indices must also use a WebGL-supported type.
  const index = geometry.getIndex();

  if (index && index.array instanceof Float64Array) {
    geometry.setIndex(
      new THREE.BufferAttribute(new Uint32Array(index.array), 1),
    );
  }
}

/**
 * Render the mesh using its original vertex colours.
 */
function MeshObject({ geometry }: { geometry: THREE.BufferGeometry }) {
  const hasVertexColors = Boolean(geometry.getAttribute("color"));

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color="white"
        vertexColors={hasVertexColors}
        roughness={0.9}
        metalness={0}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Main interactive 3D viewer.
 */
export default function MeshViewer({ url }: MeshViewerProps) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    let cancelled = false;

    async function loadMesh() {
      setGeometry(null);
      setError(null);

      try {
        // 1. Download the generated PLY mesh.
        const response = await fetch(url, {
          signal: controller.signal,
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error(`Unable to load mesh (HTTP ${response.status}).`);
        }

        const buffer = await response.arrayBuffer();

        // 2. Parse the PLY file.
        const loader = new PLYLoader();

        const loaded = loader.parse(buffer);

        const position = loaded.getAttribute("position");

        if (!position || position.count === 0) {
          loaded.dispose();

          throw new Error("The PLY file contains no valid vertices.");
        }

        // 3. IMPORTANT FIX:
        // Convert unsupported Float64Array attributes.
        convertGeometryToFloat32(loaded);

        const convertedPosition = loaded.getAttribute("position");

        console.info(
          "Mesh vertex format:",
          convertedPosition.array.constructor.name,
        );

        // 4. Calculate the geometry bounds.
        loaded.computeBoundingBox();

        const boundingBox = loaded.boundingBox;

        if (!boundingBox) {
          loaded.dispose();

          throw new Error("Could not calculate mesh dimensions.");
        }

        const center = new THREE.Vector3();
        const size = new THREE.Vector3();

        boundingBox.getCenter(center);
        boundingBox.getSize(size);

        const largestDimension = Math.max(size.x, size.y, size.z);

        if (!Number.isFinite(largestDimension) || largestDimension <= 0) {
          loaded.dispose();

          throw new Error("The mesh dimensions are invalid.");
        }

        // 5. Move the mesh centre to the origin.
        loaded.translate(-center.x, -center.y, -center.z);

        // 6. Scale the mesh to fit the viewer.
        const scale = 2.5 / largestDimension;

        // ImageTo3D uses Y-down, Z-forward coordinates.
        // Convert them to the viewer's orientation.
        loaded.scale(scale, -scale, -scale);

        // 7. Recalculate normals for lighting.
        loaded.deleteAttribute("normal");
        loaded.computeVertexNormals();

        if (cancelled) {
          loaded.dispose();
          return;
        }

        // 8. Save the prepared mesh in React state.
        setGeometry(loaded);
      } catch (cause) {
        if (cancelled) {
          return;
        }

        setError(
          cause instanceof Error
            ? cause.message
            : "Unable to display the 3D mesh.",
        );
      }
    }

    void loadMesh();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [url]);

  // Release geometry resources on replacement or unmount.
  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (error) {
    return (
      <div className="flex h-full w-full items-center justify-center p-6 text-center text-sm text-red-300">
        {error}
      </div>
    );
  }

  if (!geometry) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-slate-400">
        Loading 3D mesh...
      </div>
    );
  }

  return (
    <div className="h-full w-full overflow-hidden">
      <Canvas
        camera={{
          position: [0, 0, 4.5],
          fov: 50,
          near: 0.1,
          far: 100,
        }}
        style={{
          width: "100%",
          height: "100%",
        }}
      >
        <color attach="background" args={["#0b1220"]} />

        <ambientLight intensity={1.5} />

        <directionalLight position={[3, 4, 5]} intensity={2} />

        <directionalLight position={[-3, -2, -4]} intensity={0.7} />

        <MeshObject geometry={geometry} />

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          minDistance={1.5}
          maxDistance={12}
        />
      </Canvas>
    </div>
  );
}

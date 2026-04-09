function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalize(vec) {
  const length = Math.hypot(vec[0], vec[1], vec[2]);
  if (length < 1e-8) {
    return [0, 0, 1];
  }
  return [vec[0] / length, vec[1] / length, vec[2] / length];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(vec, scalar) {
  return [vec[0] * scalar, vec[1] * scalar, vec[2] * scalar];
}

function lookAtCamera(position, target, upDirection) {
  let forward = normalize(subtract(target, position));
  let right = cross(forward, upDirection);
  if (Math.hypot(right[0], right[1], right[2]) < 1e-8) {
    right = cross(forward, [0, 0, 1]);
  }
  right = normalize(right);
  const up = normalize(cross(right, forward));
  return [
    [right[0], up[0], forward[0], position[0]],
    [right[1], up[1], forward[1], position[1]],
    [right[2], up[2], forward[2], position[2]],
    [0, 0, 0, 1],
  ];
}

function matrixColumn(matrix, index) {
  return [matrix[0][index], matrix[1][index], matrix[2][index]];
}

function conventionRotation(cameraConvention) {
  const rotations = {
    opencv: [
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, 1],
    ],
    opengl: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, -1],
    ],
    blender: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, -1],
    ],
    colmap: [
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, 1],
    ],
  };
  return rotations[cameraConvention] ?? rotations.opencv;
}

function multiplyMat3(a, b) {
  const result = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      let sum = 0;
      for (let index = 0; index < 3; index += 1) {
        sum += a[row][index] * b[index][col];
      }
      result[row][col] = sum;
    }
  }
  return result;
}

function convertCamToWorldConvention(
  camToWorld,
  sourceConvention,
  targetConvention,
) {
  const sourceTransform = conventionRotation(sourceConvention);
  const targetTransform = conventionRotation(targetConvention);
  const rotation = [
    [camToWorld[0][0], camToWorld[0][1], camToWorld[0][2]],
    [camToWorld[1][0], camToWorld[1][1], camToWorld[1][2]],
    [camToWorld[2][0], camToWorld[2][1], camToWorld[2][2]],
  ];
  const internalRotation = multiplyMat3(rotation, sourceTransform);
  const targetRotation = multiplyMat3(internalRotation, targetTransform);
  return [
    [targetRotation[0][0], targetRotation[0][1], targetRotation[0][2], camToWorld[0][3]],
    [targetRotation[1][0], targetRotation[1][1], targetRotation[1][2], camToWorld[1][3]],
    [targetRotation[2][0], targetRotation[2][1], targetRotation[2][2], camToWorld[2][3]],
    [0, 0, 0, 1],
  ];
}

function parseCameraState(cameraStateJson) {
  const externalState = JSON.parse(cameraStateJson);
  const convention = externalState.camera_convention ?? "opencv";
  return {
    fov_degrees: externalState.fov_degrees,
    width: externalState.width,
    height: externalState.height,
    camera_convention: convention,
    cam_to_world: convertCamToWorldConvention(
      externalState.cam_to_world,
      convention,
      "opencv",
    ),
  };
}

function bufferToUint8Array(buffer) {
  if (buffer instanceof Uint8Array) {
    return buffer;
  }
  if (buffer instanceof ArrayBuffer) {
    return new Uint8Array(buffer);
  }
  if (ArrayBuffer.isView(buffer)) {
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }
  return null;
}

function parseFramePacket(packet) {
  const bytes = bufferToUint8Array(packet);
  if (bytes === null || bytes.byteLength < 4) {
    return null;
  }
  const headerLength =
    (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
  if (headerLength < 0 || bytes.byteLength < 4 + headerLength) {
    return null;
  }
  const headerBytes = bytes.subarray(4, 4 + headerLength);
  const payload = bytes.subarray(4 + headerLength);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  return { header, payload };
}

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "native-viewer-root";

  const frame = document.createElement("canvas");
  frame.className = "native-viewer-frame";
  frame.tabIndex = 0;
  frame.setAttribute("aria-label", "Native 3D viewer");
  root.appendChild(frame);
  const frameContext = frame.getContext("2d");
  if (frameContext === null) {
    throw new Error("Failed to acquire 2D canvas context.");
  }

  const overlay = document.createElement("div");
  overlay.className = "native-viewer-overlay";

  const latencyBadge = document.createElement("div");
  latencyBadge.className = "native-viewer-badge native-viewer-latency";
  latencyBadge.hidden = true;

  overlay.appendChild(latencyBadge);
  root.appendChild(overlay);
  el.appendChild(root);

  let cameraState = parseCameraState(model.get("camera_state_json"));
  let position = [
    cameraState.cam_to_world[0][3],
    cameraState.cam_to_world[1][3],
    cameraState.cam_to_world[2][3],
  ];
  let target = add(position, scale(matrixColumn(cameraState.cam_to_world, 2), 3.0));
  let orbitDistance = Math.max(1e-3, Math.hypot(...subtract(position, target)));
  let interaction = null;
  let animationFrame = null;
  let lastTickMs = null;
  const pressedKeys = new Set();
  const clickThresholdPixels = 4.0;
  let lastFrameRevision = -1;
  let averageLatencyMs = null;
  let lastLatencySampleMs = null;
  let lastLatencySampleAtMs = null;
  let lastRenderTimeMs = null;
  let lastDecodeTimeMs = null;
  let lastDrawTimeMs = null;
  let lastPresentWaitTimeMs = null;
  let lastReceiveQueueTimeMs = null;
  let lastPostReceiveTimeMs = null;
  let lastPacketSizeBytes = 0;
  let lastBackendToBrowserTimeMs = null;
  let smoothedRenderTimeMs = null;
  let smoothedDecodeTimeMs = null;
  let smoothedDrawTimeMs = null;
  let smoothedPresentWaitTimeMs = null;
  let smoothedReceiveQueueTimeMs = null;
  let smoothedPostReceiveTimeMs = null;
  let smoothedPacketSizeBytes = null;
  let smoothedBackendToBrowserTimeMs = null;
  const revisionSentAtMs = new Map();
  let latestScheduledFrameRevision = -1;
  let interactionActive = Boolean(model.get("interaction_active"));
  let settleTimeoutId = null;
  const settleDelayMs = 150;
  let streamSocket = null;
  let reconnectTimeoutId = null;
  let closed = false;
  let nextClockSyncPingId = 0;
  let bestClockSyncRttMs = null;
  let backendClockOffsetMs = null;
  const pendingClockSyncPings = new Map();

  function updateAspectRatio() {
    const explicitAspectRatio = Number(model.get("aspect_ratio"));
    const fallbackAspectRatio =
      Math.max(1, cameraState.width) / Math.max(1, cameraState.height);
    const aspectRatio =
      Number.isFinite(explicitAspectRatio) && explicitAspectRatio > 0.0
        ? explicitAspectRatio
        : fallbackAspectRatio;
    root.style.aspectRatio = `${aspectRatio}`;
  }

  function getViewportSize() {
    const rect = root.getBoundingClientRect();
    return {
      width: Math.max(1, Math.round(rect.width || cameraState.width)),
      height: Math.max(1, Math.round(rect.height || cameraState.height)),
    };
  }

  function updateLatencyBadge() {
    if (averageLatencyMs === null) {
      latencyBadge.hidden = true;
      return;
    }
    latencyBadge.hidden = false;
    const latencyText = `${Math.round(averageLatencyMs)} ms`;
    if (lastRenderTimeMs === null) {
      latencyBadge.textContent = latencyText;
      return;
    }
    latencyBadge.textContent =
      `${latencyText} | render ${Math.round(lastRenderTimeMs)} ms`;
  }

  function syncMetricsToModel() {
    if (
      averageLatencyMs === null
      || smoothedRenderTimeMs === null
      || smoothedReceiveQueueTimeMs === null
      || smoothedPostReceiveTimeMs === null
      || smoothedDecodeTimeMs === null
      || smoothedDrawTimeMs === null
      || smoothedPresentWaitTimeMs === null
      || smoothedPacketSizeBytes === null
    ) {
      return;
    }
    model.set("latency_ms", averageLatencyMs);
    model.set("latency_sample_ms", lastLatencySampleMs);
    model.set("render_time_ms", smoothedRenderTimeMs);
    model.set(
      "backend_to_browser_time_ms",
      smoothedBackendToBrowserTimeMs ?? 0.0,
    );
    model.set("packet_size_bytes", Math.round(smoothedPacketSizeBytes));
    model.set("browser_receive_queue_ms", smoothedReceiveQueueTimeMs);
    model.set("browser_post_receive_ms", smoothedPostReceiveTimeMs);
    model.set("browser_decode_time_ms", smoothedDecodeTimeMs);
    model.set("browser_draw_time_ms", smoothedDrawTimeMs);
    model.set("browser_present_wait_ms", smoothedPresentWaitTimeMs);
    model.save_changes();
  }

  function smoothMetric(previous, sample, shouldReset) {
    if (previous === null || shouldReset) {
      return sample;
    }
    return previous * 0.85 + sample * 0.15;
  }

  function updatePoseFromMatrix() {
    position = [
      cameraState.cam_to_world[0][3],
      cameraState.cam_to_world[1][3],
      cameraState.cam_to_world[2][3],
    ];
  }

  function serializeCameraState() {
    return JSON.stringify({
      fov_degrees: cameraState.fov_degrees,
      width: cameraState.width,
      height: cameraState.height,
      camera_convention: cameraState.camera_convention,
      cam_to_world: convertCamToWorldConvention(
        cameraState.cam_to_world,
        "opencv",
        cameraState.camera_convention,
      ),
    });
  }

  function updateCameraMatrix() {
    cameraState.cam_to_world = lookAtCamera(position, target, [0, 1, 0]);
  }

  function syncSizeIntoCameraState() {
    const viewport = getViewportSize();
    cameraState.width = viewport.width;
    cameraState.height = viewport.height;
  }

  function pushCameraState() {
    syncSizeIntoCameraState();
    updateCameraMatrix();
    const nextJson = serializeCameraState();
    if (nextJson === model.get("camera_state_json")) {
      return;
    }
    const nextRevision = model.get("_camera_revision") + 1;
    revisionSentAtMs.set(nextRevision, performance.now());
    model.set("camera_state_json", nextJson);
    model.set("_camera_revision", nextRevision);
    model.save_changes();
  }

  function requestSettledRender() {
    const nextRevision = model.get("_camera_revision") + 1;
    revisionSentAtMs.set(nextRevision, performance.now());
    model.set("interaction_active", false);
    model.set("_camera_revision", nextRevision);
    model.save_changes();
    interactionActive = false;
  }

  function scheduleSettledRender() {
    if (settleTimeoutId !== null) {
      clearTimeout(settleTimeoutId);
    }
    settleTimeoutId = setTimeout(() => {
      settleTimeoutId = null;
      if (interaction !== null || pressedKeys.size > 0) {
        return;
      }
      if (interactionActive) {
        requestSettledRender();
      }
    }, settleDelayMs);
  }

  function markInteractionActive() {
    if (settleTimeoutId !== null) {
      clearTimeout(settleTimeoutId);
      settleTimeoutId = null;
    }
    if (interactionActive) {
      return;
    }
    interactionActive = true;
    model.set("interaction_active", true);
    model.save_changes();
  }

  function orbit(deltaX, deltaY) {
    const rotationSpeed = 0.008;
    const offset = subtract(position, target);
    const radius = Math.max(1e-3, Math.hypot(...offset));
    const yaw = Math.atan2(offset[0], offset[2]) - deltaX * rotationSpeed;
    const pitch = clamp(
      Math.atan2(offset[1], Math.hypot(offset[0], offset[2])) - deltaY * rotationSpeed,
      -Math.PI / 2 + 1e-3,
      Math.PI / 2 - 1e-3,
    );
    const cosPitch = Math.cos(pitch);
    position = [
      target[0] + radius * Math.sin(yaw) * cosPitch,
      target[1] + radius * Math.sin(pitch),
      target[2] + radius * Math.cos(yaw) * cosPitch,
    ];
    orbitDistance = radius;
  }

  function pan(deltaX, deltaY) {
    const forward = normalize(subtract(target, position));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = normalize(cross(right, forward));
    const scaleFactor =
      Math.max(1e-3, orbitDistance) *
      Math.tan((cameraState.fov_degrees * Math.PI / 180.0) / 2.0) /
      Math.max(1, frame.getBoundingClientRect().height) *
      2.0;
    const translation = add(
      scale(right, -deltaX * scaleFactor),
      scale(up, deltaY * scaleFactor),
    );
    position = add(position, translation);
    target = add(target, translation);
  }

  function dolly(deltaY) {
    const zoomFactor = Math.exp(deltaY * 0.0015);
    const offset = subtract(position, target);
    orbitDistance = clamp(Math.hypot(...offset) * zoomFactor, 0.05, 1e5);
    const direction = normalize(offset);
    position = add(target, scale(direction, orbitDistance));
  }

  function stepKeyboard(deltaSeconds) {
    if (pressedKeys.size === 0) {
      return;
    }
    const forward = normalize(subtract(target, position));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = [0, 1, 0];
    const speed = Math.max(0.05, orbitDistance * 0.8) * deltaSeconds;
    let motion = [0, 0, 0];
    if (pressedKeys.has("w")) motion = add(motion, scale(forward, speed));
    if (pressedKeys.has("s")) motion = add(motion, scale(forward, -speed));
    if (pressedKeys.has("a")) motion = add(motion, scale(right, -speed));
    if (pressedKeys.has("d")) motion = add(motion, scale(right, speed));
    if (pressedKeys.has("q")) motion = add(motion, scale(up, -speed));
    if (pressedKeys.has("e")) motion = add(motion, scale(up, speed));
    position = add(position, motion);
    target = add(target, motion);
    pushCameraState();
  }

  function tick(timestamp) {
    if (lastTickMs === null) {
      lastTickMs = timestamp;
    }
    const deltaSeconds = Math.min(0.05, (timestamp - lastTickMs) / 1000);
    lastTickMs = timestamp;
    stepKeyboard(deltaSeconds);
    if (pressedKeys.size > 0) {
      animationFrame = requestAnimationFrame(tick);
    } else {
      animationFrame = null;
    }
  }

  function ensureKeyboardLoop() {
    if (animationFrame === null) {
      lastTickMs = null;
      animationFrame = requestAnimationFrame(tick);
    }
  }

  function registerFrameMetrics(revision, renderTimeMs, shouldReset) {
    let latencySampleMs = null;
    const sentAtMs = revisionSentAtMs.get(revision);
    if (sentAtMs !== undefined) {
      const now = performance.now();
      const latencyMs = Math.max(0.0, now - sentAtMs);
      latencySampleMs = latencyMs;
      averageLatencyMs =
        averageLatencyMs === null || shouldReset
          ? latencyMs
          : averageLatencyMs * 0.85 + latencyMs * 0.15;
      lastLatencySampleAtMs = now;
      revisionSentAtMs.delete(revision);
    }
    if (typeof renderTimeMs === "number" && Number.isFinite(renderTimeMs)) {
      lastRenderTimeMs = renderTimeMs;
    }
    for (const pendingRevision of revisionSentAtMs.keys()) {
      if (pendingRevision < revision) {
        revisionSentAtMs.delete(pendingRevision);
      }
    }
    if (latencySampleMs !== null) {
      lastLatencySampleMs = latencySampleMs;
    }
    if (typeof renderTimeMs === "number" && Number.isFinite(renderTimeMs)) {
      smoothedRenderTimeMs = smoothMetric(
        smoothedRenderTimeMs,
        renderTimeMs,
        shouldReset,
      );
    }
    updateLatencyBadge();
    syncMetricsToModel();
  }

  async function drawFrame(
    bytes,
    width,
    height,
    revision,
    renderTimeMs,
    interactionActiveFrame,
    mimeType,
    messageReceivedAtMs,
    backendFrameSentPerfTimeMs,
  ) {
    latestScheduledFrameRevision = Math.max(latestScheduledFrameRevision, revision);
    const decodeEnqueueAt = performance.now();
    const shouldReset =
      lastLatencySampleAtMs === null
      || decodeEnqueueAt - lastLatencySampleAtMs > 1000.0;
    if (revision < latestScheduledFrameRevision || revision < lastFrameRevision) {
      return;
    }
    const blob = new Blob([bytes], { type: mimeType || "image/jpeg" });
    const decodeStartedAt = performance.now();
    const bitmap = await createImageBitmap(blob);
    const decodeFinishedAt = performance.now();
    if (revision < latestScheduledFrameRevision || revision < lastFrameRevision) {
      bitmap.close();
      return;
    }
    const drawStartedAt = performance.now();
    frame.width = width;
    frame.height = height;
    frameContext.clearRect(0, 0, width, height);
    frameContext.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    lastFrameRevision = revision;
    const drawFinishedAt = performance.now();
    if (interactionActiveFrame) {
      const receiveQueueTimeMs = decodeEnqueueAt - messageReceivedAtMs;
      lastReceiveQueueTimeMs = receiveQueueTimeMs;
      lastPacketSizeBytes = bytes.byteLength;
      await new Promise((resolve) => {
        requestAnimationFrame(() => resolve(performance.now()));
      }).then((presentedAtNow) => {
        lastPresentWaitTimeMs = presentedAtNow - drawFinishedAt;
      });
      lastDecodeTimeMs = decodeFinishedAt - decodeStartedAt;
      lastDrawTimeMs = drawFinishedAt - drawStartedAt;
      lastPostReceiveTimeMs = performance.now() - messageReceivedAtMs;
      const shouldReset =
        lastLatencySampleAtMs === null
        || performance.now() - lastLatencySampleAtMs > 1000.0;
      if (
        backendClockOffsetMs !== null
        && typeof backendFrameSentPerfTimeMs === "number"
        && Number.isFinite(backendFrameSentPerfTimeMs)
      ) {
        const estimatedClientSentAtMs =
          backendFrameSentPerfTimeMs - backendClockOffsetMs;
        lastBackendToBrowserTimeMs = Math.max(
          0.0,
          messageReceivedAtMs - estimatedClientSentAtMs,
        );
        smoothedBackendToBrowserTimeMs = smoothMetric(
          smoothedBackendToBrowserTimeMs,
          lastBackendToBrowserTimeMs,
          shouldReset,
        );
      }
      smoothedDecodeTimeMs = smoothMetric(
        smoothedDecodeTimeMs,
        lastDecodeTimeMs,
        shouldReset,
      );
      smoothedDrawTimeMs = smoothMetric(
        smoothedDrawTimeMs,
        lastDrawTimeMs,
        shouldReset,
      );
      smoothedPresentWaitTimeMs = smoothMetric(
        smoothedPresentWaitTimeMs,
        lastPresentWaitTimeMs,
        shouldReset,
      );
      smoothedReceiveQueueTimeMs = smoothMetric(
        smoothedReceiveQueueTimeMs,
        receiveQueueTimeMs,
        shouldReset,
      );
      smoothedPostReceiveTimeMs = smoothMetric(
        smoothedPostReceiveTimeMs,
        lastPostReceiveTimeMs,
        shouldReset,
      );
      smoothedPacketSizeBytes = smoothMetric(
        smoothedPacketSizeBytes,
        lastPacketSizeBytes,
        shouldReset,
      );
      registerFrameMetrics(revision, renderTimeMs, shouldReset);
      return;
    }
    revisionSentAtMs.delete(revision);
  }

  function scheduleReconnect() {
    if (closed || reconnectTimeoutId !== null) {
      return;
    }
    reconnectTimeoutId = setTimeout(() => {
      reconnectTimeoutId = null;
      connectFrameStream();
    }, 250);
  }

  function sendClockSyncPing() {
    if (streamSocket === null || streamSocket.readyState !== WebSocket.OPEN) {
      return;
    }
    const pingId = nextClockSyncPingId;
    nextClockSyncPingId += 1;
    const clientSentAtMs = performance.now();
    pendingClockSyncPings.set(pingId, clientSentAtMs);
    streamSocket.send(JSON.stringify({
      type: "clock_sync_ping",
      ping_id: pingId,
      client_sent_at_ms: clientSentAtMs,
    }));
  }

  function disconnectFrameStream() {
    if (reconnectTimeoutId !== null) {
      clearTimeout(reconnectTimeoutId);
      reconnectTimeoutId = null;
    }
    pendingClockSyncPings.clear();
    if (streamSocket !== null) {
      const socket = streamSocket;
      streamSocket = null;
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      if (
        socket.readyState === WebSocket.OPEN
        || socket.readyState === WebSocket.CONNECTING
      ) {
        socket.close();
      }
    }
  }

  function connectFrameStream() {
    disconnectFrameStream();
    const streamPort = Number(model.get("stream_port"));
    const streamPath = model.get("stream_path");
    const streamToken = model.get("stream_token");
    if (!Number.isFinite(streamPort) || streamPort <= 0 || !streamPath || !streamToken) {
      return;
    }
    const streamUrl =
      `ws://${window.location.hostname}:${streamPort}${streamPath}?token=${encodeURIComponent(streamToken)}`;
    const socket = new WebSocket(streamUrl);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      sendClockSyncPing();
    };
    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        let message = null;
        try {
          message = JSON.parse(event.data);
        } catch (_error) {
          return;
        }
        if (message.type !== "clock_sync_pong") {
          return;
        }
        const pingId = Number(message.ping_id);
        const clientSentAtMs = pendingClockSyncPings.get(pingId);
        if (clientSentAtMs === undefined) {
          return;
        }
        pendingClockSyncPings.delete(pingId);
        const clientReceivedAtMs = performance.now();
        const serverReceivedAtMs = Number(message.server_received_at_ms);
        if (!Number.isFinite(serverReceivedAtMs)) {
          return;
        }
        const rttMs = clientReceivedAtMs - clientSentAtMs;
        const offsetMs =
          serverReceivedAtMs - ((clientSentAtMs + clientReceivedAtMs) / 2.0);
        if (bestClockSyncRttMs === null || rttMs < bestClockSyncRttMs) {
          bestClockSyncRttMs = rttMs;
          backendClockOffsetMs = offsetMs;
        } else if (backendClockOffsetMs === null) {
          backendClockOffsetMs = offsetMs;
        } else {
          backendClockOffsetMs = backendClockOffsetMs * 0.9 + offsetMs * 0.1;
        }
        return;
      }
      const messageReceivedAtMs = performance.now();
      const packet = parseFramePacket(event.data);
      if (packet === null) {
        return;
      }
      if (
        typeof packet.header.revision === "number"
        && packet.header.revision % 30 === 0
      ) {
        sendClockSyncPing();
      }
      void drawFrame(
        packet.payload,
        packet.header.width ?? 0,
        packet.header.height ?? 0,
        packet.header.revision ?? -1,
        packet.header.render_time_ms,
        Boolean(packet.header.interaction_active),
        packet.header.mime_type,
        messageReceivedAtMs,
        packet.header.backend_frame_sent_perf_time_ms,
      );
    };
    socket.onerror = () => {
      scheduleReconnect();
    };
    socket.onclose = () => {
      if (streamSocket === socket) {
        streamSocket = null;
      }
      scheduleReconnect();
    };
    streamSocket = socket;
  }

  function applyCameraStateJson() {
    const incoming = model.get("camera_state_json");
    if (incoming === serializeCameraState()) {
      return;
    }
    cameraState = parseCameraState(incoming);
    updatePoseFromMatrix();
    updateAspectRatio();
    const forward = matrixColumn(cameraState.cam_to_world, 2);
    target = add(position, scale(forward, orbitDistance));
  }

  frame.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  frame.addEventListener("pointerdown", (event) => {
    markInteractionActive();
    frame.focus();
    frame.setPointerCapture(event.pointerId);
    interaction = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      button: event.button,
      moved: false,
      mode: event.button === 2 ? "pan" : "orbit",
    };
    frame.classList.add("is-dragging");
  });

  frame.addEventListener("pointermove", (event) => {
    if (!interaction || interaction.pointerId !== event.pointerId) {
      return;
    }
    const deltaX = event.clientX - interaction.lastX;
    const deltaY = event.clientY - interaction.lastY;
    const dragDistance = Math.hypot(
      event.clientX - interaction.startX,
      event.clientY - interaction.startY,
    );
    if (dragDistance > clickThresholdPixels) {
      interaction.moved = true;
    }
    interaction.lastX = event.clientX;
    interaction.lastY = event.clientY;
    if (interaction.mode === "orbit") {
      orbit(deltaX, deltaY);
    } else {
      pan(deltaX, deltaY);
    }
    pushCameraState();
  });

  function endInteraction(event) {
    if (!interaction || interaction.pointerId !== event.pointerId) {
      return;
    }
    const shouldRegisterClick =
      interaction.button === 0 && !interaction.moved;
    if (shouldRegisterClick) {
      const rect = frame.getBoundingClientRect();
      const normalizedX = rect.width > 0
        ? (event.clientX - rect.left) / rect.width
        : 0.0;
      const normalizedY = rect.height > 0
        ? (event.clientY - rect.top) / rect.height
        : 0.0;
      const clickPayload = {
        x: Math.max(
          0,
          Math.min(
            cameraState.width - 1,
            Math.floor(normalizedX * cameraState.width),
          ),
        ),
        y: Math.max(
          0,
          Math.min(
            cameraState.height - 1,
            Math.floor(normalizedY * cameraState.height),
          ),
        ),
        width: cameraState.width,
        height: cameraState.height,
        camera_state: JSON.parse(serializeCameraState()),
      };
      model.set("last_click_json", JSON.stringify(clickPayload));
      model.save_changes();
    }
    frame.releasePointerCapture(event.pointerId);
    interaction = null;
    frame.classList.remove("is-dragging");
    scheduleSettledRender();
  }

  frame.addEventListener("pointerup", endInteraction);
  frame.addEventListener("pointercancel", endInteraction);

  frame.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      markInteractionActive();
      dolly(event.deltaY);
      pushCameraState();
      scheduleSettledRender();
    },
    { passive: false },
  );

  frame.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d", "q", "e"].includes(key)) {
      return;
    }
    event.preventDefault();
    markInteractionActive();
    pressedKeys.add(key);
    ensureKeyboardLoop();
  });

  frame.addEventListener("keyup", (event) => {
    pressedKeys.delete(event.key.toLowerCase());
    if (pressedKeys.size === 0) {
      scheduleSettledRender();
    }
  });

  frame.addEventListener("blur", () => {
    pressedKeys.clear();
    scheduleSettledRender();
  });

  const resizeObserver = new ResizeObserver(() => {
    pushCameraState();
  });
  resizeObserver.observe(root);

  const onCameraChange = () => applyCameraStateJson();
  const onAspectRatioChange = () => {
    updateAspectRatio();
    pushCameraState();
  };
  const onInteractionActiveChange = () => {
    interactionActive = Boolean(model.get("interaction_active"));
  };
  const onStreamConfigChange = () => {
    connectFrameStream();
  };

  model.on("change:camera_state_json", onCameraChange);
  model.on("change:aspect_ratio", onAspectRatioChange);
  model.on("change:interaction_active", onInteractionActiveChange);
  model.on("change:stream_port", onStreamConfigChange);
  model.on("change:stream_path", onStreamConfigChange);
  model.on("change:stream_token", onStreamConfigChange);

  updateAspectRatio();
  updateLatencyBadge();
  connectFrameStream();
  pushCameraState();

  return () => {
    resizeObserver.disconnect();
    if (animationFrame !== null) {
      cancelAnimationFrame(animationFrame);
    }
    closed = true;
    disconnectFrameStream();
    model.off("change:camera_state_json", onCameraChange);
    model.off("change:aspect_ratio", onAspectRatioChange);
    model.off("change:interaction_active", onInteractionActiveChange);
    model.off("change:stream_port", onStreamConfigChange);
    model.off("change:stream_path", onStreamConfigChange);
    model.off("change:stream_token", onStreamConfigChange);
    if (settleTimeoutId !== null) {
      clearTimeout(settleTimeoutId);
    }
  };
}

const widget = { render };

export { render };
export default widget;

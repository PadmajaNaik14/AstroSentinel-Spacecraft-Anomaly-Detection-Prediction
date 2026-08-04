import { useEffect, useRef, useState } from "react";
import { wsUrl } from "../api/client";
import type { TelemetryPoint } from "../types";

const RING_BUFFER_SIZE = 500;

/** Live WS telemetry stream for one channel. `paused` stops consuming new
 * messages client-side (the connection stays open, per plan.md -- resuming
 * just continues forward, no catch-up on missed points). */
export function useChannelStream(chanId: string, paused: boolean) {
  const [points, setPoints] = useState<TelemetryPoint[]>([]);
  const [connected, setConnected] = useState(false);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    setPoints([]);
    setConnected(false);
    const ws = new WebSocket(wsUrl(chanId));

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      if (pausedRef.current) return;
      const point: TelemetryPoint = JSON.parse(event.data);
      setPoints((prev) => {
        const next = prev.length >= RING_BUFFER_SIZE ? prev.slice(1) : prev;
        return [...next, point];
      });
    };

    return () => ws.close();
  }, [chanId]);

  return { points, connected };
}

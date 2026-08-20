// Innohack Traffic Congestion AI Explorer JS

const API_BASE = "http://localhost:8000";

// Fallback telemetry data
const fallbackCorridors = [
    { name: "Silk Board Junction", speed: 11.5, congestion: 92.4, status: "SEVERE", color: "#ef4444", delay: 33.0, cam: "CAM-3049" },
    { name: "Outer Ring Road (Bellandur)", speed: 17.2, congestion: 81.5, status: "HEAVY", color: "#f97316", delay: 24.5, cam: "CAM-1923" },
    { name: "MG Road Signal", speed: 22.0, congestion: 64.0, status: "MODERATE", color: "#f59e0b", delay: 12.0, cam: "CAM-0812" },
    { name: "Tin Factory (KR Puram)", speed: 9.8, congestion: 94.2, status: "SEVERE", color: "#ef4444", delay: 38.0, cam: "CAM-4120" },
    { name: "Hebbal Flyover Junction", speed: 38.5, congestion: 45.0, status: "MODERATE", color: "#f59e0b", delay: 6.5, cam: "CAM-1055" },
    { name: "Koramangala 80ft Road", speed: 28.0, congestion: 38.0, status: "CLEAR", color: "#10b981", delay: 3.2, cam: "CAM-0544" },
    { name: "Electronic City Expressway", speed: 52.0, congestion: 35.0, status: "CLEAR", color: "#10b981", delay: 2.0, cam: "CAM-7711" },
    { name: "Indiranagar 100ft Road", speed: 21.0, congestion: 58.0, status: "MODERATE", color: "#f59e0b", delay: 9.0, cam: "CAM-2219" }
];

const bmdSamples = [
    {
        name: "Silk Board CCTV (CAM-3049)",
        boxes: [
            { class: "Auto-rickshaw", color: "#f59e0b", x: 0.15, y: 0.45, w: 0.14, h: 0.22, conf: 0.94 },
            { class: "Motorbike", color: "#06b6d4", x: 0.32, y: 0.55, w: 0.08, h: 0.18, conf: 0.98 },
            { class: "BMTC Bus", color: "#ef4444", x: 0.45, y: 0.30, w: 0.25, h: 0.40, conf: 0.96 },
            { class: "Car", color: "#6366f1", x: 0.72, y: 0.50, w: 0.18, h: 0.25, conf: 0.91 },
            { class: "Auto-rickshaw", color: "#f59e0b", x: 0.05, y: 0.60, w: 0.12, h: 0.20, conf: 0.89 }
        ]
    },
    {
        name: "MG Road CCTV (CAM-0812)",
        boxes: [
            { class: "Car", color: "#6366f1", x: 0.10, y: 0.40, w: 0.20, h: 0.30, conf: 0.95 },
            { class: "LCV", color: "#10b981", x: 0.35, y: 0.35, w: 0.18, h: 0.32, conf: 0.92 },
            { class: "Motorbike", color: "#06b6d4", x: 0.55, y: 0.52, w: 0.09, h: 0.20, conf: 0.97 },
            { class: "Motorbike", color: "#06b6d4", x: 0.66, y: 0.54, w: 0.08, h: 0.19, conf: 0.94 }
        ]
    }
];

let activeSampleIdx = 0;

document.addEventListener("DOMContentLoaded", () => {
    initCorridorsGrid(fallbackCorridors);
    drawCCTVCanvas();
    startLiveTick();
});

// Render Corridor Telemetry List
function initCorridorsGrid(corridors) {
    const listEl = document.getElementById("corridorsList");
    if (!listEl) return;

    listEl.innerHTML = "";
    corridors.forEach(corr => {
        const row = document.createElement("div");
        row.className = "corridor-row";
        row.innerHTML = `
            <div class="corr-info">
                <h4>${corr.name}</h4>
                <p>Speed: ${corr.speed} km/h • Delay: +${corr.delay} mins • ${corr.cam}</p>
            </div>
            <div class="status-tag" style="background: ${corr.color}20; color: ${corr.color}; border: 1px solid ${corr.color}40;">
                ${corr.status} (${corr.congestion}%)
            </div>
        `;
        listEl.appendChild(row);
    });
}

// Render CCTV Bounding Box Overlay
function drawCCTVCanvas() {
    const canvas = document.getElementById("cctvCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const w = 720;
    const h = 405;
    canvas.width = w;
    canvas.height = h;

    const sample = bmdSamples[activeSampleIdx];

    // Draw Simulated CCTV Frame Background
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, w, h);

    // Draw Road Surface
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(15, 23, 42, 0.85)");
    grad.addColorStop(1, "rgba(30, 41, 59, 0.95)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Draw Lane Markings
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 2;
    for (let i = 0; i < w; i += 50) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i + (i - w/2)*0.4, h);
        ctx.stroke();
    }

    // Draw Bounding Boxes
    sample.boxes.forEach(b => {
        const bx = b.x * w;
        const by = b.y * h;
        const bw = b.w * w;
        const bh = b.h * h;

        ctx.fillStyle = b.color + "25";
        ctx.fillRect(bx, by, bw, bh);

        ctx.strokeStyle = b.color;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(bx, by, bw, bh);

        const lbl = `${b.class} ${(b.conf * 100).toFixed(0)}%`;
        ctx.font = "600 11px Inter, sans-serif";
        const tw = ctx.measureText(lbl).width;

        ctx.fillStyle = b.color;
        ctx.fillRect(bx, by - 20, tw + 10, 20);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(lbl, bx + 5, by - 6);
    });

    // HUD Text
    ctx.fillStyle = "#10b981";
    ctx.font = "600 12px Outfit, sans-serif";
    ctx.fillText(`● REC [${sample.name}] 1920x1080 @ 30 FPS`, 14, 24);
    ctx.fillText(`BMD-45 ANNOTATION MODEL ACTIVE`, 14, 42);
}

function switchSample() {
    activeSampleIdx = (activeSampleIdx + 1) % bmdSamples.length;
    drawCCTVCanvas();
}

// Emergency Ambulance Green Wave Trigger
function triggerEmergency() {
    const banner = document.getElementById("emergencyBanner");
    if (banner) {
        banner.style.display = "flex";
        banner.scrollIntoView({ behavior: 'smooth' });
    }
}

function dismissEmergency() {
    const banner = document.getElementById("emergencyBanner");
    if (banner) {
        banner.style.display = "none";
    }
}

// Periodic live sensor telemetry tick simulation
function startLiveTick() {
    setInterval(() => {
        fallbackCorridors.forEach(c => {
            const delta = (Math.random() - 0.5) * 1.5;
            c.congestion = Math.min(98.0, Math.max(10.0, +(c.congestion + delta).toFixed(1)));
        });
        initCorridorsGrid(fallbackCorridors);
    }, 4000);
}

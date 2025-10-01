import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { setupPWAInstall } from "./utils/pwaInstall";

// Register service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .catch(() => {
        // Service worker registration failed, continuing without offline support
      });
  });
}

// Setup PWA install prompt
setupPWAInstall();

createRoot(document.getElementById("root")!).render(<App />);

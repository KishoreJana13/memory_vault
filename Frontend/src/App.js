import React, { useState } from "react";
import { BrowserRouter as Router, Route, Link, Routes, Navigate, useNavigate } from "react-router-dom";
import "./App.css";
import Home from "./components/Home"; 
import QueryPage from "./components/QueryPage";
import MemoryInput from "./components/MemoryInput";

const RecallMe = () => {
  const RecallApp = React.lazy(() => import('./recall/App'));

  return (
    <div className="recall-container">
      <React.Suspense fallback={<div>Loading RecallMe...</div>}>
        <RecallApp />
      </React.Suspense>
    </div>
  );
};

function App() {
  const [showRecallApp, setShowRecallApp] = useState(false);

  const handleRecallClick = (e) => {
    e.preventDefault();
    setShowRecallApp(true);
  };

  const handleMemoryVaultClick = () => {
    setShowRecallApp(false);
  };

  if (showRecallApp) {
    return (
      <div className="App">
        <header className="App-header">
          <nav className="nav-menu">
            <div className="nav-links-left">
              <button 
                onClick={handleMemoryVaultClick} 
                className="nav-link"
                id="backToMemoryVault"
              >
                Back to MemoryVault
              </button>
            </div>
            <div className="logo-title-container">
              <img
                src="/logo.png"
                alt="MemoryVault Logo"
                className="nav-logo"
                onClick={handleMemoryVaultClick} 
                style={{ cursor: "pointer" }}
              />
              <span 
                className="nav-title" 
                id="recallMeTitle"
                onClick={handleMemoryVaultClick}
                style={{ cursor: "pointer" }}
              >
                RecallMe
              </span>
            </div>
            <div className="nav-links-right">
              <div className="nav-link-placeholder"></div>
            </div>
          </nav>
        </header>

        <RecallMe />
      </div>
    );
  }

  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <nav className="nav-menu">
            <div className="nav-links-left">
              <Link to="/query-memories" className="nav-link" id="queryMemories">
                Query Memories
              </Link>
              <Link to="/add-memory" className="nav-link" id="addMemory">
                Add Memory
              </Link>
            </div>
            <div className="logo-title-container">
              <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center" }}>
                <img
                  src="/logo.png"
                  alt="MemoryVault Logo"
                  className="nav-logo"
                  style={{ cursor: "pointer" }}
                />
                <span className="nav-title" id="memoryVault" style={{ cursor: "pointer" }}>
                  MemoryVault
                </span>
              </Link>
            </div>
            <div className="nav-links-right">
              <a 
                href="#" 
                onClick={handleRecallClick} 
                className="nav-link" 
                id="recallMe"
              >
                RecallMe
              </a>
            </div>
          </nav>
        </header>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/query-memories" element={<QueryPage />} />
          <Route path="/add-memory" element={<MemoryInput />} />
          <Route path="/query" element={<Navigate to="/query-memories" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

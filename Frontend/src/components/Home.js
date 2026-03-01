

import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import "./Home.css";

const Home = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  // Add these images for your memory bubbles
  const memoryImages = [
    "/images/image1.jpg",
    "/images/image2.jpg",
    "/images/image3.jpg"
];
  return (
    <div className="home-container">
      <motion.div 
        className="hero-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: isVisible ? 1 : 0, y: isVisible ? 0 : 20 }}
        transition={{ duration: 0.8 }}
      >
        <h1>
          <span className="title-highlight">Memory</span>Vault
        </h1>
        <p className="tagline">Preserving precious memories for those with Alzheimer's and dementia</p>
        
        <div className="memory-illustration">
          <div className="memory-bubble memory-bubble-1">
            <img src={memoryImages[0]} alt="Family member 1" className="memory-image" />
          </div>
          <div className="memory-bubble memory-bubble-2">
            <img src={memoryImages[1]} alt="Family member 2" className="memory-image" />
          </div>
          <div className="memory-bubble memory-bubble-3">
            <img src={memoryImages[2]} alt="Family gathering" className="memory-image" />
          </div>
        </div>
        
        <div className="cta-buttons">
          <Link to="/add-memory" className="cta-button primary">
            <span className="cta-icon">✨</span>
            Start Adding Memories
          </Link>
          <Link to="/query-memories" className="cta-button secondary">
            <span className="cta-icon">🔍</span>
            Query Existing Memories
          </Link>
        </div>
      </motion.div>
      
      <motion.div 
        className="features-section"
        initial={{ opacity: 0 }}
        animate={{ opacity: isVisible ? 1 : 0 }}
        transition={{ duration: 0.8, delay: 0.3 }}
      >
        <div className="feature-card">
          <div className="feature-icon conversation-icon">
            <span>💬</span>
          </div>
          <h3>Natural Conversations</h3>
          <p>Engage in natural conversations about past experiences through our advanced AI interface</p>
        </div>
        
        <div className="feature-card">
          <div className="feature-icon family-icon">
            <span>👪</span>
          </div>
          <h3>Family Collaboration</h3>
          <p>Invite family members to contribute memories, creating a collaborative space for preservation</p>
        </div>
        
        <div className="feature-card">
          <div className="feature-icon wellbeing-icon">
            <span>❤️</span>
          </div>
          <h3>Emotional Wellbeing</h3>
          <p>Strengthen emotional connections and improve wellbeing through memory recollection</p>
        </div>
      </motion.div>
      
      <motion.div 
        className="unique-features-section"
        initial={{ opacity: 0 }}
        animate={{ opacity: isVisible ? 1 : 0 }}
        transition={{ duration: 0.8, delay: 0.6 }}
      >
        <h2>Unique Memory-Preserving Features</h2>
        <div className="unique-features-container">
          <div className="unique-feature">
            <div className="unique-feature-icon">👤</div>
            <div className="unique-feature-content">
              <h3>Facial Recognition</h3>
              <p>Instantly identify family members using camera recognition, eliminating the need for reintroductions</p>
            </div>
          </div>
          
          <div className="unique-feature">
            <div className="unique-feature-icon">🔊</div>
            <div className="unique-feature-content">
              <h3>Voice Messages</h3>
              <p>Listen to recorded messages from loved ones, reinforcing familiarity through voice recognition</p>
            </div>
          </div>
          
          <div className="unique-feature">
            <div className="unique-feature-icon">🧠</div>
            <div className="unique-feature-content">
              <h3>    Memory Retrieval System</h3>
              <p>Smart AI connects and presents relevant memories based on conversation context.</p>
            </div>
          </div>
        </div>
      </motion.div>
      
      <motion.div 
        className="how-it-works"
        initial={{ opacity: 0 }}
        animate={{ opacity: isVisible ? 1 : 0 }}
        transition={{ duration: 0.8, delay: 0.9 }}
      >
        <h2>How MemoryVault Works</h2>
        <div className="steps-container">
          <div className="step">
            <div className="step-number">1</div>
            <h4>Add Memories</h4>
            <p>Record important memories, events, and relationships with text</p>
            <div className="step-icon">📸</div>
          </div>
          
          <div className="step">
            <div className="step-number">2</div>
            <h4>AI Processing</h4>
            <p>Our AI organizes and connects memories for natural recall and personalized experiences</p>
            <div className="step-icon">🧠</div>
          </div>
          
          <div className="step">
            <div className="step-number">3</div>
            <h4>Query & Recall</h4>
            <p>Ask questions to retrieve memories in a conversational way</p>
            <div className="step-icon">💭</div>
          </div>
        </div>
        
        <div className="steps-container second-row">
        <div className="step">
          <div className="step-number">4</div>
          <h4>Capture Images</h4>
          <p>Take photos instantly using the camera with facial recognition, making reintroductions unnecessary.</p>
          <div className="step-icon">📷</div>
        </div>

        <div className="step">
          <div className="step-number">5</div>
          <h4>Family Photos</h4>
          <p>Store predefined photos of family members and loved ones to help retrieve memories.</p>
          <div className="step-icon">👨‍👩‍👧‍👦</div>
        </div>

        <div className="step">
          <div className="step-number">6</div>
          <h4>Member Information</h4>
          <p>Photos include key details about relationships, life events, and recorded audio from that person.</p>
          <div className="step-icon">📋</div>
        </div>
      </div>

      </motion.div>
      
      <motion.div 
        className="get-started-section"
        initial={{ opacity: 0 }}
        animate={{ opacity: isVisible ? 1 : 0 }}
        transition={{ duration: 0.8, delay: 1.5 }}
      >
        <h2>Start Preserving Memories Today</h2>
        <p>Helping families stay connected and preserve precious memories.  
        Let loved ones relive cherished moments with MemoryVault.</p>
        <Link to="/query-memories" className="cta-button primary large">
          Begin Your Memory Journey
        </Link>
      </motion.div>
    </div>
  );
};

export default Home;
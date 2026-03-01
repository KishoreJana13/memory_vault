

import React, { useState, useRef, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { photos } from "./Photos"; // Import the photos array
import { FaFileAlt, FaPlay, FaPause, FaChevronLeft, FaChevronRight, FaHeart, FaDownload, FaShare } from "react-icons/fa";
import "./Audiophoto.css";

function Audiophoto() {
  const { photoId } = useParams();
  const currentPhotoId = Number(photoId);
  const navigate = useNavigate();
  
  // Find the current photo
  const photo = photos.find(p => p.id === currentPhotoId);
  
  // Filter photos for navigation
  const memberPhotos = photos;
  const currentPhotoIndex = memberPhotos.findIndex(p => p.id === currentPhotoId);
  
  const [transcription, setTranscription] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [showTranscription, setShowTranscription] = useState(true);
  const [isFavorite, setIsFavorite] = useState(false);
  const [audioProgress, setAudioProgress] = useState(0);
  const audioRef = useRef(null);
  
  // Map photo IDs to audio files
  const audioMap = {
    1: process.env.PUBLIC_URL + "/audio/family_message1.wav",
    2: process.env.PUBLIC_URL + "/audio/family_message2.wav",
    3: process.env.PUBLIC_URL + "/audio/family_message3.wav",
    4: process.env.PUBLIC_URL + "/audio/family_message4.wav",
    5: process.env.PUBLIC_URL + "/audio/family_message5.wav",
    6: process.env.PUBLIC_URL + "/audio/family_message6.wav",
    7: process.env.PUBLIC_URL + "/audio/family_message7.wav",
    8: process.env.PUBLIC_URL + "/audio/family_message8.wav",
    9: process.env.PUBLIC_URL + "/audio/family_message9.wav",
    10: process.env.PUBLIC_URL + "/audio/family_message10.wav",
    11: process.env.PUBLIC_URL + "/audio/family_message11.wav",
  };
  
  // Fallback transcriptions for each photo
  const fallbackTranscriptions = {
    1: "First time at Harris' house—felt like a VIP. Then, Pavan, Harris, and I hit 'I Love Kovai,' acting like travel vloggers. Ended up chilling near the boat house, cracking dumb jokes like philosophers. Easily one of those days you don't forget.",
    2: "Jeep ride in Coorg with the boys—pure cinema. Cool breeze, scenic hills, and us screaming like stunt doubles in an action movie. Every bump had us airborne, holding on for dear life. Harris, of course, sat like a boss, unbothered. Absolute madness, absolute fun.",
    3: "Post-presentation, we collapsed onto the bus—zombies with swollen eyes. Nobody spoke, just nodded off like bobbleheads. If we missed our stops, so be it. That kind of exhaustion hits different after a solid day.",
    4: "You love sugarcane, so you went straight to the fields like a man on a mission. Found the perfect one, took a bite, and instantly felt like a village hero. Chewing away, enjoying the sweet taste, while lowkey hoping no one thought I was stealing. Totally worth it!",
    5: "We walked into Bannari Amman Institute like Nobel Prize winners—white shirts, full confidence, zero fear. Talked like we invented something revolutionary (did we? No idea). No clue if people understood, but we walked out like legends.",
    6: "Won a cricket trophy—felt like a one-man army. Held it up like a Bollywood hero, expecting background music. Instead, just silence and a couple of confused stares. No team, no parade, just me and my trophy… but hey, a win's a win!",
    7: "First time eating shawarma with Pavan and Harris—felt like a pro until I heard the word mayo. Immediate gag, dignity gone. Meanwhile, Harris, the shawarma assassin, casually devoured his and my share like a legend. We laughed, we suffered, and Harris walked away like a king.",
    8: "First time in our placement suits—felt like billionaires, bank balance said otherwise. Walked around like runway models, fixing our ties every two minutes. One of us even pulled out a fake business card. Zero offers, 100% swagger.",
    9: "Cultural fest at college—energy, music, and Bavya in a saree, looking like she stepped out of a movie scene. Meanwhile, you were struggling with your outfit, feeling like an overgrown school kid.",
    10: "Bavya went to the flower shop and came back with a bright red flower—just to congratulate you on winning the hackathon.",
    11: "Bavya came home to decorate your house for your birthday. She put up balloons and decorations while you just watched. After that, you laughed, ate a lot, and had a great time. It was a perfect birthday for you!"
  };

  // Navigate to next or previous photo
  const navigateToPhoto = (direction) => {
    let newIndex = currentPhotoIndex + direction;
    
    // Handle wrap-around
    if (newIndex < 0) {
      newIndex = memberPhotos.length - 1;
    } else if (newIndex >= memberPhotos.length) {
      newIndex = 0;
    }
    
    // Navigate to the new photo
    navigate(`/photo/${memberPhotos[newIndex].id}`);
  };

  // Set the initial transcription when the component mounts or photoId changes
  useEffect(() => {
    if (photoId && fallbackTranscriptions[photoId]) {
      setTranscription(fallbackTranscriptions[photoId]);
    }
    
    // Check if photo is in favorites
    const favorites = JSON.parse(localStorage.getItem('favorites') || '[]');
    setIsFavorite(favorites.includes(Number(photoId)));
    
    // Reset audio progress when changing photos
    setAudioProgress(0);
    setIsPlaying(false);
  }, [photoId]);

  // Preload the audio based on photoId
  useEffect(() => {
    if (photo && audioRef.current) {
      audioRef.current.src = audioMap[photoId] || "";
      
      // Add event listeners
      const handleEnded = () => setIsPlaying(false);
      const handleError = (e) => {
        console.error("Audio error:", e);
        setError("Failed to load audio file");
      };
      const updateProgress = () => {
        if (audioRef.current) {
          const progress = (audioRef.current.currentTime / audioRef.current.duration) * 100;
          setAudioProgress(progress || 0);
        }
      };
      
      audioRef.current.addEventListener("ended", handleEnded);
      audioRef.current.addEventListener("error", handleError);
      audioRef.current.addEventListener("timeupdate", updateProgress);
    
      return () => {
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current.removeEventListener("ended", handleEnded);
          audioRef.current.removeEventListener("error", handleError);
          audioRef.current.removeEventListener("timeupdate", updateProgress);
        }
      };
    }
  }, [photoId, photo]);

  const handlePlayPause = () => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      // Play the audio
      audioRef.current.play()
        .then(() => {
          setIsPlaying(true);
        })
        .catch(err => {
          console.error("Failed to play audio:", err);
          setError("Failed to play audio. Please try again.");
        });
    }
  };
  
  const toggleFavorite = () => {
    const favorites = JSON.parse(localStorage.getItem('favorites') || '[]');
    let newFavorites;
    
    if (isFavorite) {
      newFavorites = favorites.filter(id => id !== Number(photoId));
    } else {
      newFavorites = [...favorites, Number(photoId)];
    }
    
    localStorage.setItem('favorites', JSON.stringify(newFavorites));
    setIsFavorite(!isFavorite);
  };
  
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = photo.src;
    link.download = `memory_${photoId}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: `Memory`,
        text: transcription,
        url: window.location.href,
      })
      .catch(err => console.error("Share failed:", err));
    } else {
      navigator.clipboard.writeText(window.location.href)
        .then(() => alert("Link copied to clipboard!"))
        .catch(err => console.error("Copy failed:", err));
    }
  };

  if (!photo) {
    return (
      <div className="error-container">
        <h1 className="error-message">Photo not found!</h1>
      </div>
    );
  }

  return (
    <div className="audiophoto-container">
      <div className="header">
        <div className="memory-counter">
          <span className="counter-text">
            {memberPhotos.length > 0 && `Memory ${currentPhotoIndex + 1}/${memberPhotos.length}`}
          </span>
        </div>
      </div>
      
      <div className="content-wrapper">
        {memberPhotos.length > 1 && (
          <button className="navigation-button left" onClick={() => navigateToPhoto(-1)}>
            <FaChevronLeft />
          </button>
        )}
        
        <div className="image-container">
          <img 
            src={photo.src} 
            alt={`Memory ${currentPhotoId}`} 
            className="memory-image"
          />
          
          <div className="image-actions">
            <button 
              className={`action-button ${isFavorite ? 'favorite-active' : ''}`} 
              onClick={toggleFavorite}
              aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
            >
              <FaHeart />
            </button>
            <button className="action-button" onClick={handleDownload} aria-label="Download image">
              <FaDownload />
            </button>
            <button className="action-button" onClick={handleShare} aria-label="Share memory">
              <FaShare />
            </button>
          </div>
        </div>
        
        {memberPhotos.length > 1 && (
          <button className="navigation-button right" onClick={() => navigateToPhoto(1)}>
            <FaChevronRight />
          </button>
        )}
      </div>
      
      <div className="controls-container">
        {/* Hidden audio element */}
        <audio ref={audioRef} preload="auto" />
        
        <div className="audio-player">
          <button 
            className={`play-button ${isPlaying ? 'playing' : ''}`}
            onClick={handlePlayPause}
            disabled={!audioMap[photoId]}
            aria-label={isPlaying ? "Pause audio" : "Play audio"}
          >
            {isPlaying ? <FaPause /> : <FaPlay />}
          </button>
          
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${audioProgress}%` }}
              ></div>
            </div>
            <span className="progress-label">
              {isPlaying ? "Now Playing" : "Play Memory Audio"}
            </span>
            {error && <span className="error-text">{error}</span>}
          </div>
        </div>
        
        <div className="transcription-container">
          <div 
            className="transcription-header" 
            onClick={() => setShowTranscription(!showTranscription)}
          >
            <div className="header-icon">
              <FaFileAlt />
            </div>
            <h3>Memory Transcription</h3>
            <span className="toggle-icon">{showTranscription ? '−' : '+'}</span>
          </div>
          
          {showTranscription && (
            <div className="transcription-content">
              {isLoading ? (
                <p className="loading-text">Loading transcription...</p>
              ) : (
                <p>{transcription || "No transcription available for this photo."}</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Audiophoto;


import React, { useState } from 'react';
import { Link } from "react-router-dom";
import photo1 from './profile_memories/monish/pic1.jpg';
import photo2 from './profile_memories/monish/pic2.jpg';
import photo3 from './profile_memories/monish/pic3.jpg';
import photo4 from './profile_memories/monish/pic4.jpg';
import photo5 from './profile_memories/monish/pic5.jpg';
import photo6 from './profile_memories/monish/pic6.jpg';
import photo7 from './profile_memories/monish/pic7.jpg';
import photo8 from './profile_memories/monish/pic8.jpg';
import photo9 from './profile_memories/monish/pic9.jpg';
import photo10 from './profile_memories/monish/pic10.jpg';
import photo11 from './profile_memories/monish/pic11.jpg';
import './index.css';

export const photos = [
  { id: 1, src: photo1 },
  { id: 2, src: photo2 },
  { id: 3, src: photo3 },
  { id: 4, src: photo4 },
  { id: 5, src: photo5 },
  { id: 6, src: photo6 },
  { id: 7, src: photo7 },
  { id: 8, src: photo8 },
  { id: 9, src: photo9 },
  { id: 10, src: photo10 },
  { id: 11, src: photo11 },
];

function User() {
  const [activePhoto, setActivePhoto] = useState(1); // Set default to 1 instead of null
  const [isGridView, setIsGridView] = useState(false); // Start with carousel view since that's where the issue is

  const handlePhotoClick = (id) => {
    setActivePhoto(id === activePhoto ? null : id);
  };

  const toggleView = () => {
    setIsGridView(!isGridView);
    setActivePhoto(isGridView ? 1 : null); // Set to 1 when switching to carousel
  };

  return (
    <div className="photo-gallery-container">
      <div className="gallery-header">
        <h1 className="gallery-title">Memories Collection</h1>
        <p className="gallery-subtitle">Click on a photo to explore</p>
        <button className="view-toggle-btn" onClick={toggleView}>
          {isGridView ? "Switch to Carousel" : "Switch to Grid"}
        </button>
      </div>

      {isGridView ? (
        <div className="photo-grid">
          {photos.map((photo) => (
            <div 
              key={photo.id} 
              className={`photo-item ${activePhoto === photo.id ? 'active' : ''}`}
              onClick={() => handlePhotoClick(photo.id)}
            >
              <div className="photo-wrapper">
                <img src={photo.src} alt={`Memory ${photo.id}`} className="photo-img" />
                <div className="photo-overlay">
                  <span className="photo-number">{photo.id}</span>
                  <Link to={`/photo/${photo.id}`} className="view-details-btn">
                    View Details
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="carousel-container">
          <button className="carousel-nav prev-btn" onClick={() => {
            if (activePhoto > 1) {
              setActivePhoto(activePhoto - 1);
            } else {
              setActivePhoto(photos.length);
            }
          }}>❮</button>
          
          <div className="carousel-track">
            {photos.map((photo) => (
              <div 
                key={photo.id} 
                className={`carousel-slide ${activePhoto === photo.id ? 'active' : ''}`}
                style={{
                  transform: activePhoto === photo.id ? 'scale(1)' : 'scale(0.8)',
                  opacity: activePhoto === photo.id ? 1 : 0,
                  visibility: activePhoto === photo.id ? 'visible' : 'hidden',
                  zIndex: activePhoto === photo.id ? 5 : 0
                }}
              >
                <img src={photo.src} alt={`Memory ${photo.id}`} className="carousel-img" />
                <div className="carousel-caption">
                  <Link to={`/photo/${photo.id}`} className="carousel-details-btn">
                    Explore Memory #{photo.id}
                  </Link>
                </div>
              </div>
            ))}
          </div>
          
          <button className="carousel-nav next-btn" onClick={() => {
            if (activePhoto < photos.length) {
              setActivePhoto(activePhoto + 1);
            } else {
              setActivePhoto(1);
            }
          }}>❯</button>
          
          <div className="carousel-thumbnails">
            {photos.map((photo) => (
              <div 
                key={photo.id} 
                className={`carousel-thumbnail ${activePhoto === photo.id ? 'active' : ''}`}
                onClick={() => setActivePhoto(photo.id)}
              >
                <img src={photo.src} alt={`Thumbnail ${photo.id}`} />
              </div>
            ))}
          </div>
        </div>
      )}
      
      <style jsx>{`
        .photo-gallery-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem;
          font-family: 'Poppins', sans-serif;
          background-color: #f8f9fa;
          min-height: 100vh;
        }
        
        .gallery-header {
          text-align: center;
          margin-bottom: 2rem;
        }
        
        .gallery-title {
          font-size: 2.5rem;
          color: #333;
          margin-bottom: 0.5rem;
          font-weight: 700;
        }
        
        .gallery-subtitle {
          color: #666;
          font-size: 1.1rem;
          margin-bottom: 1.5rem;
        }
        
        .view-toggle-btn {
          background-color: #4a6cf7;
          color: white;
          border: none;
          padding: 0.75rem 1.5rem;
          border-radius: 30px;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.3s ease;
          font-weight: 600;
        }
        
        .view-toggle-btn:hover {
          background-color: #3451b2;
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(74, 108, 247, 0.3);
        }
        
        /* Grid View Styles */
        .photo-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 1.5rem;
          margin-top: 2rem;
        }
        
        .photo-item {
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
          cursor: pointer;
          transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
          position: relative;
        }
        
        .photo-item:hover {
          transform: translateY(-10px);
          box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
        }
        
        .photo-item.active {
          transform: translateY(-10px) scale(1.05);
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
          z-index: 2;
        }
        
        .photo-wrapper {
          position: relative;
          overflow: hidden;
          height: 0;
          padding-bottom: 100%; /* 1:1 Aspect Ratio */
        }
        
        .photo-img {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: transform 0.5s ease;
        }
        
        .photo-item:hover .photo-img {
          transform: scale(1.1);
        }
        
        .photo-overlay {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          background: linear-gradient(to top, rgba(0, 0, 0, 0.8), transparent);
          padding: 1.5rem;
          opacity: 0;
          transform: translateY(20px);
          transition: all 0.3s ease;
        }
        
        .photo-item:hover .photo-overlay,
        .photo-item.active .photo-overlay {
          opacity: 1;
          transform: translateY(0);
        }
        
        .photo-number {
          position: absolute;
          top: 15px;
          right: 15px;
          background-color: #4a6cf7;
          color: white;
          width: 30px;
          height: 30px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          font-size: 0.9rem;
        }
        
        .view-details-btn {
          display: inline-block;
          background-color: #fff;
          color: #333;
          padding: 0.5rem 1rem;
          border-radius: 25px;
          text-decoration: none;
          font-size: 0.9rem;
          font-weight: 600;
          transition: all 0.3s ease;
        }
        
        .view-details-btn:hover {
          background-color: #4a6cf7;
          color: #fff;
          transform: translateY(-2px);
        }
        
        /* Carousel View Styles */
        .carousel-container {
          position: relative;
          padding: 2rem 0;
          height: 70vh;
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        
        .carousel-track {
          display: flex;
          justify-content: center;
          align-items: center;
          width: 100%;
          height: 80%;
          position: relative;
          overflow: hidden;
        }
        
        .carousel-slide {
          position: absolute;
          width: 70%;
          height: 100%;
          transition: all 0.5s cubic-bezier(0.25, 1, 0.5, 1);
          display: flex;
          justify-content: center;
          align-items: center;
          overflow: hidden;
          border-radius: 15px;
          box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
          visibility: hidden;
          opacity: 0;
        }
        
        .carousel-slide.active {
          visibility: visible;
          opacity: 1;
          z-index: 5;
        }
        
        .carousel-img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        
        .carousel-caption {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          background: linear-gradient(to top, rgba(0, 0, 0, 0.8), transparent);
          padding: 2rem;
          text-align: center;
          opacity: 0;
          transform: translateY(20px);
          transition: all 0.3s ease;
        }
        
        .carousel-slide.active .carousel-caption {
          opacity: 1;
          transform: translateY(0);
        }
        
        .carousel-details-btn {
          display: inline-block;
          background-color: #4a6cf7;
          color: white;
          padding: 0.75rem 1.5rem;
          border-radius: 30px;
          text-decoration: none;
          font-size: 1rem;
          font-weight: 600;
          transition: all 0.3s ease;
        }
        
        .carousel-details-btn:hover {
          background-color: #fff;
          color: #4a6cf7;
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(255, 255, 255, 0.3);
        }
        
        .carousel-nav {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          background-color: rgba(255, 255, 255, 0.7);
          border: none;
          width: 50px;
          height: 50px;
          border-radius: 50%;
          font-size: 1.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          z-index: 10;
          transition: all 0.3s ease;
        }
        
        .carousel-nav:hover {
          background-color: #4a6cf7;
          color: white;
        }
        
        .prev-btn {
          left: 20px;
        }
        
        .next-btn {
          right: 20px;
        }
        
        .carousel-thumbnails {
          display: flex;
          justify-content: center;
          gap: 10px;
          margin-top: 20px;
          overflow-x: auto;
          padding: 10px 0;
        }
        
        .carousel-thumbnail {
          width: 60px;
          height: 60px;
          border-radius: 8px;
          overflow: hidden;
          cursor: pointer;
          transition: all 0.3s ease;
          opacity: 0.6;
        }
        
        .carousel-thumbnail img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        
        .carousel-thumbnail:hover {
          opacity: 0.9;
          transform: translateY(-3px);
        }
        
        .carousel-thumbnail.active {
          border: 3px solid #4a6cf7;
          opacity: 1;
        }
        
        /* Responsive Adjustments */
        @media (max-width: 768px) {
          .photo-grid {
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 1rem;
          }
          
          .carousel-slide {
            width: 85%;
          }
          
          .carousel-thumbnails {
            width: 100%;
            justify-content: flex-start;
          }
        }
      `}</style>
    </div>
  );
}

export default User;
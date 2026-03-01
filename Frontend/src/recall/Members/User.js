
import React from 'react';
import { Link } from "react-router-dom";

import photo4 from '../profile_memories/monish/pic4.jpg';
import photo6 from '../profile_memories/monish/pic5.jpg';
import photo8 from '../profile_memories/monish/pic8.jpg';

function User() {
  const photos = [
    { id: 4, src: photo4 },
    { id: 6, src: photo6 },
    { id: 8, src: photo8 },
  ];

  return (
    <div className="user-profile">
      <div className="user-header">
        <div className="user-greeting">
          <h1>This is you, <span className="highlight-name">Monish</span>. Hello!</h1>
          <div className="wave-animation">👋</div>
        </div>
        
        <div className="user-bio">
          <p>You are a 21-year-old college student with a passion for movies and cricket. Your days are filled with learning and laughter,
          and you always find time to bond with your close ones—Kishore and Kaushik, who are more like the family you choose. Your favorite
          color is green, a reminder of growth, balance, and the vibrant energy you bring to every challenge.</p>
        </div>
      </div>

      <div className="interests-container">
        <div className="interest-pill"><span className="interest-icon">🎬</span> Movies</div>
        <div className="interest-pill"><span className="interest-icon">🏏</span> Cricket</div>
        <div className="interest-pill"><span className="interest-icon">📚</span> Learning</div>
        <div className="interest-pill" style={{backgroundColor: 'rgba(0, 128, 0, 0.2)'}}><span className="interest-icon">💚</span> Green</div>
      </div>

      <h2 className="gallery-heading">Your Photo Gallery</h2>

      <div className="photo-gallery-container">
        {photos.map((photo) => (
          <Link key={photo.id} to={`/photo/${photo.id}`} className="photo-card">
            <div className="photo-frame">
              <img src={photo.src} alt={`Monish memory ${photo.id}`} className="gallery-image" />
            </div>
            <div className="photo-overlay">
              <span className="view-text">View Memory</span>
            </div>
          </Link>
        ))}
      </div>

      <style jsx>{`
        /* Custom CSS for User Component */
        .user-profile {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem;
          font-family: 'Poppins', sans-serif;
          color: #333;
          background-color: #f9f9f9;
          border-radius: 20px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
          overflow: hidden;
          position: relative;
        }

        .user-profile::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 10px;
          background: linear-gradient(90deg, #4CAF50, #8BC34A, #CDDC39);
          z-index: 1;
        }

        .user-header {
          margin-bottom: 2.5rem;
          position: relative;
        }

        .user-greeting {
          display: flex;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .user-greeting h1 {
          font-size: 2.5rem;
          font-weight: 700;
          color: #2E7D32;
          margin: 0;
          line-height: 1.2;
          letter-spacing: -0.5px;
        }

        .highlight-name {
          position: relative;
          display: inline-block;
          color: #2E7D32;
          font-weight: 800;
          z-index: 1;
        }

        .highlight-name::after {
          content: '';
          position: absolute;
          bottom: 2px;
          left: 0;
          width: 100%;
          height: 8px;
          background-color: rgba(76, 175, 80, 0.3);
          z-index: -1;
          border-radius: 4px;
        }

        .wave-animation {
          font-size: 2.5rem;
          margin-left: 15px;
          animation: wave 1.5s infinite;
        }

        @keyframes wave {
          0% { transform: rotate(0deg); }
          25% { transform: rotate(20deg); }
          50% { transform: rotate(0deg); }
          75% { transform: rotate(20deg); }
          100% { transform: rotate(0deg); }
        }

        .user-bio {
          position: relative;
          padding: 1.5rem;
          background-color: white;
          border-radius: 12px;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
          line-height: 1.6;
          font-size: 1.1rem;
          border-left: 4px solid #4CAF50;
        }

        .user-bio p {
          margin: 0;
          color: #424242;
        }

        .interests-container {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-bottom: 2.5rem;
        }

        .interest-pill {
          display: flex;
          align-items: center;
          padding: 8px 16px;
          background-color: rgba(76, 175, 80, 0.1);
          border-radius: 50px;
          color: #2E7D32;
          font-weight: 600;
          font-size: 0.95rem;
          transition: all 0.3s ease;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .interest-pill:hover {
          transform: translateY(-3px);
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
          background-color: rgba(76, 175, 80, 0.2);
        }

        .interest-icon {
          margin-right: 6px;
          font-size: 1.1rem;
        }

        .gallery-heading {
          font-size: 1.8rem;
          font-weight: 700;
          color: #2E7D32;
          margin: 2rem 0 1.5rem;
          position: relative;
          display: inline-block;
        }

        .gallery-heading::after {
          content: '';
          position: absolute;
          bottom: -5px;
          left: 0;
          width: 50%;
          height: 3px;
          background: linear-gradient(90deg, #4CAF50, transparent);
          border-radius: 3px;
        }

        .photo-gallery-container {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 20px;
          margin-top: 2rem;
        }

        .photo-card {
          position: relative;
          overflow: hidden;
          border-radius: 12px;
          transition: all 0.3s ease;
          cursor: pointer;
          height: 250px;
          text-decoration: none;
          background-color: white;
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }

        .photo-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        }

        .photo-card:hover .photo-frame {
          transform: scale(1.05);
        }

        .photo-card:hover .photo-overlay {
          opacity: 1;
        }

        .photo-frame {
          width: 100%;
          height: 100%;
          transition: transform 0.5s ease;
          overflow: hidden;
        }

        .gallery-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: filter 0.3s ease;
        }

        .photo-card:hover .gallery-image {
          filter: brightness(0.7);
        }

        .photo-overlay {
          position: absolute;
          bottom: 0;
          left: 0;
          width: 100%;
          padding: 15px;
          background: linear-gradient(to top, rgba(0,0,0,0.7), transparent);
          color: white;
          opacity: 0;
          transition: opacity 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .view-text {
          background-color: rgba(76, 175, 80, 0.9);
          padding: 8px 15px;
          border-radius: 20px;
          font-weight: 600;
          font-size: 0.85rem;
          transform: translateY(10px);
          transition: transform 0.3s ease;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }

        .photo-card:hover .view-text {
          transform: translateY(0);
        }

        /* Media Queries */
        @media (max-width: 768px) {
          .user-profile {
            padding: 1.5rem;
            border-radius: 15px;
          }

          .user-greeting h1 {
            font-size: 1.8rem;
          }

          .wave-animation {
            font-size: 1.8rem;
          }

          .photo-gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
          }
          
          .user-bio {
            padding: 1rem;
            font-size: 1rem;
          }
        }

        @media (max-width: 480px) {
          .user-profile {
            padding: 1.2rem;
          }

          .user-greeting h1 {
            font-size: 1.5rem;
          }

          .photo-gallery-container {
            grid-template-columns: 1fr;
          }
          
          .interest-pill {
            padding: 6px 12px;
            font-size: 0.85rem;
          }
        }
      `}</style>
    </div>
  );
}

export default User;
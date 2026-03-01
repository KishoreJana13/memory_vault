
import React from 'react';
import { Link } from "react-router-dom";
import photo9 from '../profile_memories/bavya/pic1.jpg';
import photo10 from '../profile_memories/bavya/pic2.jpg';
import photo11 from '../profile_memories/bavya/pic3.jpg';

function Member1() {
  const photos = [
    { id: 9, src: photo9 },
    { id: 10, src: photo10 },
    { id: 11, src: photo11 },
  ];

  return (
    <div className="bavya-profile">
      <div className="profile-header">
        <div className="friend-greeting">
          <h1>This is <span className="highlight-name">Kaushik</span>, your Best Friend from Another Mother!</h1>
          <div className="heart-animation">💜</div>
        </div>
        
        <div className="friend-bio">
          <p>Kaushik is 21 years old and the kind of friend who feels like family. He loves to read and paint, and the two of you 
          can spend hours watching movies together, debating which one is better! Whether it's sharing laughs, deep talks, or 
          just goofing around, he's always there like a sibling you chose yourself.</p>
        </div>
      </div>

      <div className="interests-container">
        <div className="interest-pill"><span className="interest-icon">📚</span> Reading</div>
        <div className="interest-pill"><span className="interest-icon">🎨</span> Painting</div>
        <div className="interest-pill"><span className="interest-icon">🎬</span> Movies</div>
        <div className="interest-pill"><span className="interest-icon">🗣️</span> Deep Talks</div>
      </div>

      <h2 className="gallery-heading">Kaushik's Gallery</h2>

      <div className="photo-gallery-container">
        {photos.map((photo, index) => (
          <Link key={photo.id} to={`/photo/${photo.id}`} className="photo-card">
            <div className="photo-frame">
              <img src={photo.src} alt={`Memory with Bavya ${index + 1}`} className="gallery-image" />
            </div>
            <div className="photo-overlay">
              <span className="view-text">View Memory</span>
            </div>
          </Link>
        ))}
      </div>

      <style jsx>{`
        /* Custom CSS for Bavya Profile */
        .bavya-profile {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem;
          font-family: 'Poppins', sans-serif;
          color: #333;
          background-color: #f9f7ff;
          border-radius: 20px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
          overflow: hidden;
          position: relative;
        }

        .bavya-profile::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 10px;
          background: linear-gradient(90deg, #8E44AD, #9B59B6, #D6A4E3);
          z-index: 1;
        }

        .profile-header {
          margin-bottom: 2.5rem;
          position: relative;
        }

        .friend-greeting {
          display: flex;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .friend-greeting h1 {
          font-size: 2.5rem;
          font-weight: 700;
          color: #6A1B9A;
          margin: 0;
          line-height: 1.2;
          letter-spacing: -0.5px;
        }

        .highlight-name {
          position: relative;
          display: inline-block;
          color: #8E44AD;
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
          background-color: rgba(155, 89, 182, 0.3);
          z-index: -1;
          border-radius: 4px;
        }

        .heart-animation {
          font-size: 2.5rem;
          margin-left: 15px;
          animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.2); }
          100% { transform: scale(1); }
        }

        .friend-bio {
          position: relative;
          padding: 1.5rem;
          background-color: white;
          border-radius: 12px;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
          line-height: 1.6;
          font-size: 1.1rem;
          border-left: 4px solid #9B59B6;
        }

        .friend-bio p {
          margin: 0;
          color: #424242;
        }

        .friend-bio::before {
          content: '"';
          position: absolute;
          top: -15px;
          left: 15px;
          font-size: 60px;
          color: rgba(155, 89, 182, 0.2);
          font-family: Georgia, serif;
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
          background-color: rgba(155, 89, 182, 0.1);
          border-radius: 50px;
          color: #6A1B9A;
          font-weight: 600;
          font-size: 0.95rem;
          transition: all 0.3s ease;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .interest-pill:hover {
          transform: translateY(-3px);
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
          background-color: rgba(155, 89, 182, 0.2);
        }

        .interest-icon {
          margin-right: 6px;
          font-size: 1.1rem;
        }

        .gallery-heading {
          font-size: 1.8rem;
          font-weight: 700;
          color: #8E44AD;
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
          background: linear-gradient(90deg, #8E44AD, transparent);
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
          transform: translateY(-5px) rotate(1deg);
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
          position: relative;
        }

        .photo-frame::after {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          border: 8px solid white;
          pointer-events: none;
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
          background-color: rgba(155, 89, 182, 0.9);
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
          .bavya-profile {
            padding: 1.5rem;
            border-radius: 15px;
          }

          .friend-greeting h1 {
            font-size: 1.8rem;
          }

          .heart-animation {
            font-size: 1.8rem;
          }

          .photo-gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
          }
          
          .friend-bio {
            padding: 1rem;
            font-size: 1rem;
          }
        }

        @media (max-width: 480px) {
          .bavya-profile {
            padding: 1.2rem;
          }

          .friend-greeting h1 {
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

export default Member1;
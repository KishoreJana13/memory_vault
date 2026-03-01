
import React from 'react';
import { Link } from "react-router-dom";
import photo1 from '../profile_memories/harris/pic10.jpg';
import photo2 from '../profile_memories/harris/pic6.jpg';
import photo3 from '../profile_memories/harris/pic8.jpg';
import photo5 from '../profile_memories/harris/pic9.jpg';
import photo7 from '../profile_memories/harris/pic7.jpg';

function Member2() {
  const photos = [
    { id: 1, src: photo1 },
    { id: 2, src: photo2 },
    { id: 3, src: photo3 },
    { id: 5, src: photo5 },
    { id: 7, src: photo7 },
  ];

  return (
    <div className="harris-profile">
      <div className="profile-header">
        <div className="friend-greeting">
          <h1>This is <span className="highlight-name">Kishore</span>, Your Brother from Another Mother!</h1>
          <div className="fist-animation">👊</div>
        </div>
        
        <div className="friend-bio">
          <p>Kishore is 22 years old and more than just a friend—he's the brother you never knew you needed! He's a gaming pro, 
          a fitness freak, and always up for a challenge. Whether it's intense football matches, binge-watching your favorite 
          shows, or just pulling each other's legs, life's never boring when he's around!</p>
        </div>
      </div>

      <div className="interests-container">
        <div className="interest-pill"><span className="interest-icon">🎮</span> Gaming</div>
        <div className="interest-pill"><span className="interest-icon">💪</span> Fitness</div>
        <div className="interest-pill"><span className="interest-icon">⚽</span> Football</div>
        <div className="interest-pill"><span className="interest-icon">🍿</span> Binge-watching</div>
        <div className="interest-pill"><span className="interest-icon">🏆</span> Challenges</div>
      </div>

      <h2 className="gallery-heading">Kishore's Gallery</h2>

      <div className="photo-gallery-container">
        {photos.map((photo, index) => (
          <Link key={photo.id} to={`/photo/${photo.id}`} className="photo-card">
            <div className="photo-frame">
              <img src={photo.src} alt={`Memory with Harris ${index + 1}`} className="gallery-image" />
            </div>
            <div className="photo-overlay">
              <span className="view-text">View Memory</span>
            </div>
          </Link>
        ))}
      </div>

      <style jsx>{`
        /* Custom CSS for Harris Profile */
        .harris-profile {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem;
          font-family: 'Poppins', sans-serif;
          color: #333;
          background-color: #f5faff;
          border-radius: 20px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
          overflow: hidden;
          position: relative;
        }

        .harris-profile::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 10px;
          background: linear-gradient(90deg, #1E88E5, #42A5F5, #90CAF9);
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
          color: #0D47A1;
          margin: 0;
          line-height: 1.2;
          letter-spacing: -0.5px;
        }

        .highlight-name {
          position: relative;
          display: inline-block;
          color: #1976D2;
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
          background-color: rgba(33, 150, 243, 0.3);
          z-index: -1;
          border-radius: 4px;
        }

        .fist-animation {
          font-size: 2.5rem;
          margin-left: 15px;
          animation: bump 1.5s infinite;
          transform-origin: center;
        }

        @keyframes bump {
          0% { transform: rotate(0deg) scale(1); }
          25% { transform: rotate(20deg) scale(1.1); }
          50% { transform: rotate(0deg) scale(1); }
          75% { transform: rotate(-20deg) scale(1.1); }
          100% { transform: rotate(0deg) scale(1); }
        }

        .friend-bio {
          position: relative;
          padding: 1.5rem;
          background-color: white;
          border-radius: 12px;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
          line-height: 1.6;
          font-size: 1.1rem;
          border-left: 4px solid #1976D2;
        }

        .friend-bio p {
          margin: 0;
          color: #424242;
        }

        .friend-bio::before {
          content: '🔥';
          position: absolute;
          top: -15px;
          left: 15px;
          font-size: 30px;
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
          background-color: rgba(33, 150, 243, 0.1);
          border-radius: 50px;
          color: #0D47A1;
          font-weight: 600;
          font-size: 0.95rem;
          transition: all 0.3s ease;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .interest-pill:hover {
          transform: translateY(-3px);
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
          background-color: rgba(33, 150, 243, 0.2);
        }

        .interest-icon {
          margin-right: 6px;
          font-size: 1.1rem;
        }

        .gallery-heading {
          font-size: 1.8rem;
          font-weight: 700;
          color: #1976D2;
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
          background: linear-gradient(90deg, #1976D2, transparent);
          border-radius: 3px;
        }

        .photo-gallery-container {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
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
          transform: perspective(800px) rotateY(0deg);
        }

        .photo-card:hover {
          transform: perspective(800px) rotateY(5deg) translateY(-5px);
          box-shadow: -10px 15px 30px rgba(0, 0, 0, 0.15);
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

        .photo-frame::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: linear-gradient(135deg, rgba(33, 150, 243, 0.2) 0%, rgba(66, 165, 245, 0) 50%);
          z-index: 1;
          pointer-events: none;
        }

        .gallery-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: filter 0.3s ease;
        }

        .photo-card:hover .gallery-image {
          filter: brightness(0.8) contrast(1.1);
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
          z-index: 2;
        }

        .view-text {
          background-color: rgba(25, 118, 210, 0.9);
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
          .harris-profile {
            padding: 1.5rem;
            border-radius: 15px;
          }

          .friend-greeting h1 {
            font-size: 1.8rem;
          }

          .fist-animation {
            font-size: 1.8rem;
          }

          .photo-gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 15px;
          }
          
          .friend-bio {
            padding: 1rem;
            font-size: 1rem;
          }
        }

        @media (max-width: 480px) {
          .harris-profile {
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

export default Member2;
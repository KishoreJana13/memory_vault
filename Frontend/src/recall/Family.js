
import React from 'react';
import { Link } from 'react-router-dom';
import './Family.css';

const Family = () => {
  return (
    <div className="family-container">
      <h1 className="title">👨‍👩‍👧‍👦 My Family</h1>
      <div className="family-grid">
        <Link to="/100" className="family-card">
          <img src="/images/image2.jpg" alt="Monish" className="family-image" />
          <div className="family-info">
            <h2>Monish </h2>
          </div>
        </Link>
        <Link to="/102" className="family-card">
          <img src="/images/image3.jpg" alt="Bavya" className="family-image" />
          <div className="family-info">
            <h2>Kaushik</h2>
          </div>
        </Link>
        <Link to="/101" className="family-card">
          <img src="/images/image1.jpg" alt="Harris" className="family-image" />
          <div className="family-info">
            <h2>Kishore</h2>
          </div>
        </Link>
      </div>
    </div>
  );
};

export default Family;


import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './Navbar';
import Family from './Family';
import Identify from './FaceRecognition';
import Photo from './Photos';
import Member1 from './Members/Member1';
import Member2 from './Members/Member2';
import User from './Members/User';
import Audiophoto from './Audiophoto';
import FamilyGallery from './FamilyGallery';
import UnknownFaces from './UnknownFaces';
import './index.css'

import Photos from './Photos';
import FaceRecognition from './FaceRecognition';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex' }}>
        <Navbar />
        <div style={{ flex: 1, padding: '20px' }}>
          <Routes>
            <Route path="/family" element={<Family />} />
            <Route path="/identify" element={<Identify />} />
            <Route path="/photos" element={<Photo />} />
            <Route path="/102" element={<Member1 />} />
            <Route path="/members" element={<FamilyGallery />} />
            <Route path="/101" element={<Member2 />} />
            <Route path="/100" element={<User />} />
            <Route path="/" element={<Photo />} />
            <Route path="/photo/:photoId" element={<Audiophoto />} />
            <Route path="/unknown-faces" element={<UnknownFaces />} />

            <Route path="/identify" element={<FaceRecognition />} />
            <Route path="/photos" element={<Photos />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
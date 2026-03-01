
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './index.css';

function UnknownFaces() {
  const [unknownImages, setUnknownImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  
  const API_URL = 'http://localhost:8000';
  
  useEffect(() => {
    fetchUnknownImages();
  }, []);
  
  const fetchUnknownImages = async () => {
    try {
      setLoading(true);
      // Log the API URL we're trying to access
      console.log("Fetching from:", `${API_URL}/get-unknown-images/`);
      
      // Try to get the directory listing manually
      const response = await axios.get(`${API_URL}/get-unknown-images/`);
      console.log("Response data:", response.data);
      
      setUnknownImages(response.data.images || []);
      setError(null);
    } catch (err) {
      console.error('Error fetching unknown images:', err);
      
      // Try an alternative approach - directly list files in the directory
      try {
        // First check if we can access the health endpoint
        const healthCheck = await axios.get(`${API_URL}/health/`);
        console.log("Health check:", healthCheck.data);
        
        setError('Backend is running but unknown images endpoint returned an error. Check server logs.');
      } catch (healthErr) {
        setError('Failed to connect to backend server. Please ensure the server is running.');
      }
    } finally {
      setLoading(false);
    }
  };
  
  const deleteImage = async (filename) => {
    if (window.confirm('Are you sure you want to delete this image?')) {
      try {
        await axios.delete(`${API_URL}/delete-unknown-image/${filename}`);
        // Refresh the list after deletion
        fetchUnknownImages();
        if (selectedImage === filename) {
          setSelectedImage(null);
        }
      } catch (err) {
        console.error('Error deleting image:', err);
        setError('Failed to delete image. Please try again.');
      }
    }
  };
  
  // Function to display images from local file system
  const getImageUrl = (imageName) => {
    // Try using a direct file path if the backend method fails
    return `${API_URL}/get-unknown-image/${imageName}`;
  };

  return (
    <div className="unknown-faces-container">
      <div className="header-section">
        <h1 className="title">Unknown Faces</h1>
        <p className="subtitle">These are faces that RecallMe couldn't identify. Review them to help improve recognition.</p>
      </div>
      
      {loading ? (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading unknown faces...</p>
        </div>
      ) : error ? (
        <div className="error-container">
          <div className="error-icon">!</div>
          <div className="error-content">
            <h3>Connection Error</h3>
            <p>{error}</p>
            <button 
              className="retry-button"
              onClick={fetchUnknownImages}
            >
              <span className="retry-icon">↻</span> Retry
            </button>
          </div>
        </div>
      ) : unknownImages.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">👁️</div>
          <p>No unknown faces found.</p>
        </div>
      ) : (
        <div className="content-container">
          <div className="image-grid">
            {unknownImages.map((imageName) => (
              <div 
                key={imageName} 
                className={`image-card ${selectedImage === imageName ? 'selected' : ''}`}
                onClick={() => setSelectedImage(selectedImage === imageName ? null : imageName)}
              >
                <div className="image-wrapper">
                  <img
                    src={getImageUrl(imageName)}
                    alt={`Unknown face: ${imageName}`}
                    onError={(e) => {
                      console.error(`Failed to load image: ${imageName}`);
                      e.target.src = 'https://via.placeholder.com/200?text=Image+Not+Found';
                    }}
                  />
                </div>
                <div className="image-info">
                  <span className="image-name">{imageName}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteImage(imageName);
                    }}
                    className="delete-button"
                    aria-label="Delete image"
                  >
                    <span className="delete-icon">×</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
          
          {selectedImage && (
            <div className="image-preview">
              <div className="preview-container">
                <img
                  src={getImageUrl(selectedImage)}
                  alt={`Selected unknown face: ${selectedImage}`}
                  className="preview-image"
                />
                <div className="preview-actions">
                  <h3>{selectedImage}</h3>
                  <button
                    onClick={() => deleteImage(selectedImage)}
                    className="delete-button-large"
                  >
                    Delete
                  </button>
                </div>
              </div>
              <button 
                className="close-preview" 
                onClick={() => setSelectedImage(null)}
              >
                ×
              </button>
            </div>
          )}
        </div>
      )}
      
      <style jsx>{`
        .unknown-faces-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem;
          font-family: 'Inter', sans-serif;
          background-color: #f9fafb;
          min-height: 100vh;
          color: #1f2937;
        }
        
        .header-section {
          text-align: center;
          margin-bottom: 3rem;
          position: relative;
        }
        
        .header-section::after {
          content: '';
          position: absolute;
          bottom: -15px;
          left: 50%;
          transform: translateX(-50%);
          width: 100px;
          height: 4px;
          background: linear-gradient(90deg, #6366f1, #8b5cf6);
          border-radius: 2px;
        }
        
        .title {
          font-size: 2.5rem;
          font-weight: 700;
          color: #111827;
          margin-bottom: 0.5rem;
          letter-spacing: -0.025em;
        }
        
        .subtitle {
          font-size: 1.1rem;
          color: #6b7280;
          max-width: 600px;
          margin: 0 auto;
          line-height: 1.6;
        }
        
        /* Loading State */
        .loading-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 300px;
        }
        
        .loading-spinner {
          width: 50px;
          height: 50px;
          border: 4px solid rgba(99, 102, 241, 0.2);
          border-radius: 50%;
          border-top-color: #6366f1;
          animation: spin 1s ease-in-out infinite;
          margin-bottom: 1rem;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        /* Error State */
        .error-container {
          background-color: #fee2e2;
          border-radius: 12px;
          padding: 2rem;
          display: flex;
          align-items: center;
          max-width: 700px;
          margin: 0 auto;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .error-icon {
          width: 50px;
          height: 50px;
          border-radius: 50%;
          background-color: #ef4444;
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.5rem;
          font-weight: bold;
          margin-right: 1.5rem;
          flex-shrink: 0;
        }
        
        .error-content {
          flex: 1;
        }
        
        .error-content h3 {
          font-size: 1.25rem;
          color: #b91c1c;
          margin-top: 0;
          margin-bottom: 0.5rem;
        }
        
        .retry-button {
          display: inline-flex;
          align-items: center;
          background-color: #b91c1c;
          color: white;
          padding: 0.5rem 1rem;
          border-radius: 6px;
          border: none;
          font-weight: 500;
          cursor: pointer;
          margin-top: 1rem;
          transition: all 0.2s ease;
        }
        
        .retry-button:hover {
          background-color: #991b1b;
          transform: translateY(-2px);
        }
        
        .retry-icon {
          margin-right: 0.5rem;
          font-size: 1.1rem;
        }
        
        /* Empty State */
        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 300px;
          background-color: white;
          border-radius: 12px;
          padding: 3rem;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .empty-icon {
          font-size: 3rem;
          margin-bottom: 1rem;
          opacity: 0.6;
        }
        
        /* Image Grid */
        .content-container {
          position: relative;
        }
        
        .image-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 1.5rem;
        }
        
        .image-card {
          background-color: white;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          cursor: pointer;
          transition: all 0.3s ease;
          position: relative;
        }
        
        .image-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .image-card.selected {
          transform: scale(1.03);
          box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
          border: 2px solid #6366f1;
        }
        
        .image-wrapper {
          height: 0;
          padding-bottom: 100%;
          position: relative;
          overflow: hidden;
        }
        
        .image-wrapper img {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: transform 0.5s ease;
        }
        
        .image-card:hover .image-wrapper img {
          transform: scale(1.05);
        }
        
        .image-info {
          padding: 0.75rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          background-color: white;
        }
        
        .image-name {
          font-size: 0.875rem;
          font-weight: 500;
          color: #374151;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 150px;
        }
        
        .delete-button {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          border: none;
          background-color: #f3f4f6;
          color: #6b7280;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all 0.2s ease;
          padding: 0;
          font-size: 1rem;
        }
        
        .delete-button:hover {
          background-color: #ef4444;
          color: white;
        }
        
        .delete-icon {
          font-size: 1.1rem;
          line-height: 1;
        }
        
        /* Image Preview */
        .image-preview {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: rgba(0, 0, 0, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 2rem;
        }
        
        .preview-container {
          background-color: white;
          border-radius: 12px;
          overflow: hidden;
          max-width: 90%;
          max-height: 90%;
          display: flex;
          flex-direction: column;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
          animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
        
        .preview-image {
          max-width: 100%;
          max-height: 70vh;
          object-fit: contain;
        }
        
        .preview-actions {
          padding: 1.5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          background-color: white;
        }
        
        .preview-actions h3 {
          margin: 0;
          font-size: 1.25rem;
          color: #111827;
          word-break: break-all;
        }
        
        .delete-button-large {
          background-color: #ef4444;
          color: white;
          border: none;
          padding: 0.5rem 1.25rem;
          border-radius: 6px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .delete-button-large:hover {
          background-color: #dc2626;
        }
        
        .close-preview {
          position: absolute;
          top: 1rem;
          right: 1rem;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background-color: rgba(255, 255, 255, 0.2);
          border: none;
          color: white;
          font-size: 1.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .close-preview:hover {
          background-color: rgba(255, 255, 255, 0.4);
          transform: rotate(90deg);
        }
        
        /* Responsive Adjustments */
        @media (max-width: 768px) {
          .unknown-faces-container {
            padding: 1rem;
          }
          
          .image-grid {
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 1rem;
          }
          
          .title {
            font-size: 2rem;
          }
          
          .preview-actions {
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
          }
          
          .preview-actions h3 {
            font-size: 1rem;
          }
        }
      `}</style>
    </div>
  );
}

export default UnknownFaces;
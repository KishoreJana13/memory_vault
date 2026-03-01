import React, { useState } from 'react';
import { Link } from 'react-router-dom';

function Navbar() {
  const [showPopup, setShowPopup] = useState(false);

  const togglePopup = () => {
    setShowPopup(!showPopup);
  };

  const closePopup = () => {
    setShowPopup(false);
  };

  return (
    <>
      <nav style={{
        width: '250px',
        background: 'linear-gradient(135deg, #F9EDF8 0%, #F5E6FF 100%)',
        height: '100vh',
        borderRight: '1px solid #EADCE8',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        boxShadow: '2px 0 10px rgba(0,0,0,0.05)',
      }}>
        <div 
          onClick={togglePopup}
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '20px 15px',
            cursor: 'pointer',
            borderBottom: '1px solid rgba(138, 58, 143, 0.1)',
          }}
        >
          <div style={{
            background: 'linear-gradient(135deg, #9C42A1 0%, #6E3CBC 100%)',
            borderRadius: '12px',
            width: '40px',
            height: '40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginRight: '12px',
            boxShadow: '0 4px 8px rgba(110, 60, 188, 0.2)',
          }}>
            <span style={{ fontSize: '24px', color: 'white' }}>🧠</span>
          </div>
          <h1 style={{ 
            color: '#8A3A8F', 
            fontSize: '24px',
            margin: 0,
            fontWeight: 'bold'
          }}>RecallMe</h1>
        </div>

        <ul style={{
          listStyleType: 'none',
          padding: '10px',
          margin: 0,
          display: 'flex',
          flexDirection: 'column',
          gap: '5px',
        }}>
          {[
            { to: "/family", icon: "👨‍👩‍👧‍👦", label: "Family", color: "#9C42A1" },
            { to: "/identify", icon: "🔍", label: "Identify", color: "#6E3CBC" },
            { to: "/photos", icon: "📷", label: "Photos", color: "#E94CA1" },
            { to: "/unknown-faces", icon: "❓", label: "Unknown Faces", color: "#38B7C6" }
          ].map((item, index) => (
            <li key={index}>
              <Link to={item.to} style={{
                fontSize: '16px',
                color: '#734B78',
                textDecoration: 'none',
                display: 'flex',
                alignItems: 'center',
                padding: '12px',
                borderRadius: '10px',
                transition: 'all 0.3s',
                fontWeight: '500',
              }} 
              onMouseOver={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(138, 58, 143, 0.1)';
                e.currentTarget.style.transform = 'translateX(3px)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.transform = 'translateX(0)';
              }}>
                <div style={{
                  background: `linear-gradient(135deg, ${item.color} 0%, ${item.color}AA 100%)`,
                  width: '36px',
                  height: '36px',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginRight: '12px',
                  boxShadow: '0 3px 6px rgba(0,0,0,0.1)',
                }}>
                  <span style={{ fontSize: '18px', color: 'white' }}>{item.icon}</span>
                </div>
                {item.label}
              </Link>
            </li>
          ))}
        </ul>
        
        
      </nav>

      {showPopup && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.6)',
          backdropFilter: 'blur(4px)',
          zIndex: 1000,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}>
          <div style={{
            backgroundColor: 'white',
            width: '90%',
            maxWidth: '800px',
            borderRadius: '16px',
            overflow: 'hidden',
            boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)',
            animation: 'fadeIn 0.3s ease-out',
          }}>
            <div style={{
              background: 'linear-gradient(135deg, #6E3CBC 0%, #9C42A1 100%)',
              color: 'white',
              padding: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <div style={{
                  width: '50px',
                  height: '50px',
                  backgroundColor: 'rgba(255, 255, 255, 0.2)',
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginRight: '15px',
                  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                }}>
                  <span style={{ fontSize: '28px' }}>🧠</span>
                </div>
                <div>
                  <h2 style={{ margin: 0, fontWeight: 'bold' }}>Welcome to RecallMe</h2>
                  <p style={{ margin: '5px 0 0 0', fontSize: '14px', opacity: 0.9 }}>
                    Helping dementia patients reconnect with memories and loved ones
                  </p>
                </div>
              </div>
              <button 
                onClick={closePopup}
                style={{
                  background: 'rgba(255, 255, 255, 0.2)',
                  border: 'none',
                  color: 'white',
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '18px',
                  cursor: 'pointer',
                  transition: 'background 0.2s',
                }}
                onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)'}
                onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)'}
              >
                ×
              </button>
            </div>
            
            <div style={{ padding: '20px' }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                gap: '15px',
                marginBottom: '25px',
              }}>
                {[
                  {
                    icon: "👨‍👩‍👧‍👦",
                    title: "Family Connections",
                    desc: "Dedicated pages for each family member containing photos and personalized audio messages.",
                    color: "#9C42A1"
                  },
                  {
                    icon: "🎤",
                    title: "Voice Recognition",
                    desc: "Listen to recorded messages from family members, with their own voice.",
                    color: "#E94CA1"
                  },
                  {
                    icon: "👁️",
                    title: "Facial Recognition",
                    desc: "Use the camera to identify people and connect them to their personal page automatically.",
                    color: "#6E3CBC"
                  },
                  {
                    icon: "📚",
                    title: "Memory Albums",
                    desc: "Store and organize important photos in curated albums to reinforce memories.",
                    color: "#38B7C6"
                  }
                ].map((feature, index) => (
                  <div key={index} style={{
                    background: 'linear-gradient(135deg, #f8f9fa 0%, #f5f5f5 100%)',
                    borderRadius: '12px',
                    padding: '20px',
                    textAlign: 'left',
                    display: 'flex',
                    transition: 'transform 0.2s, box-shadow 0.2s',
                    boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.transform = 'translateY(-3px)';
                    e.currentTarget.style.boxShadow = '0 8px 15px rgba(0,0,0,0.1)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = '0 2px 10px rgba(0,0,0,0.05)';
                  }}>
                    <div style={{
                      width: '50px',
                      height: '50px',
                      background: `linear-gradient(135deg, ${feature.color} 0%, ${feature.color}AA 100%)`,
                      borderRadius: '10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      marginRight: '15px',
                      flexShrink: 0,
                      boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                    }}>
                      <span style={{ fontSize: '24px', color: 'white' }}>{feature.icon}</span>
                    </div>
                    <div>
                      <h3 style={{ margin: '0 0 8px 0', color: feature.color, fontWeight: 'bold' }}>{feature.title}</h3>
                      <p style={{ margin: 0, color: '#555', fontSize: '14px', lineHeight: '1.4' }}>
                        {feature.desc}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              
              <button 
                onClick={closePopup}
                style={{
                  background: 'linear-gradient(135deg, #6E3CBC 0%, #9C42A1 100%)',
                  color: 'white',
                  border: 'none',
                  padding: '14px 30px',
                  borderRadius: '30px',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  display: 'block',
                  margin: '0 auto',
                  boxShadow: '0 4px 15px rgba(110, 60, 188, 0.3)',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 6px 20px rgba(110, 60, 188, 0.4)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 15px rgba(110, 60, 188, 0.3)';
                }}
              >
                Get Started
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default Navbar;
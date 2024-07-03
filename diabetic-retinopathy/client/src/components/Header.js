import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header = ({ isAuthenticated, handleSignOut }) => {
    return (
        <header className="header">
            <nav>
              <div className="nav-links">
                    <Link to="/about" className="nav-link">About</Link>
                    {isAuthenticated && <Link to="/upload" className="nav-link">Analysis</Link>}
                    <Link to="/contact" className="nav-link">Contact</Link>
                    {isAuthenticated && <button className="sign-out-btn" onClick={handleSignOut}>Sign Out</button>}
              </div>
            </nav>
        </header>
    );
};

export default Header;

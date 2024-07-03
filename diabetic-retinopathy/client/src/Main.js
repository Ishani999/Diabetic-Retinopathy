import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import App from './App';
import Upload from './components/upload';
import Dashboard from './components/Dashboard';
import Header from './components/Header';
import Footer from './components/Footer';
import AboutPage from './components/AboutPage';
import Contact from './components/contact'; // Ensure the path is correct
import HistoryPage from './components/HistoryPage';

function Main() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    // Function to handle successful sign-in
    const handleSignIn = () => {
        setIsAuthenticated(true);
    };

    // Function to handle sign-out
    const handleSignOut = () => {
        setIsAuthenticated(false);
    };

    // Route guard function to restrict access to private routes
    const PrivateRoute = ({ element, path }) => {
        return isAuthenticated ? element : <Navigate to="/about" />;
    };

    return (
        <Router>
            <div className="App">
                <Header isAuthenticated={isAuthenticated} handleSignOut={handleSignOut} />
                <Routes>
                    <Route path="/" element={<App />} />
                    <Route path="/about" element={<AboutPage handleSignIn={handleSignIn} />} />
                    <Route path="/contact" element={<Contact />} />
                    {/* Private routes */}
                    <Route path="/upload" element={<PrivateRoute element={<Upload />} />} />
                    
                    <Route path="/history" element={<PrivateRoute element={<HistoryPage />} />} />
                    <Route path="/dashboard" element={<PrivateRoute element={<Dashboard />} />} />
                </Routes>
                <Footer />
            </div>
        </Router>
    );
}

export default Main;

import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import retinaImage from '../assests/image1.png'; 
import './AboutPage.css'; 
import axios from 'axios';
import { jwtDecode } from 'jwt-decode';

const AboutPage = ({ handleSignIn }) => {
    const navigate = useNavigate();

    useEffect(() => {
        /* global google */
        google.accounts.id.initialize({
            client_id: "41122690496-56u8g711vqh0h5afagp1lts5bch9r8hk.apps.googleusercontent.com",
            callback: handleCallbackResponse
        });

        google.accounts.id.renderButton(
            document.getElementById("signInDiv"),
            { theme: "filled_blue", size: "large",prompt: "select_account", authuser: 0 }
        );

        /*function handleCallbackResponse(response) {
            console.log("Encoded JWT ID token: " + response.credential);
            handleSignIn(); // Notify Main.js about successful sign-in
            navigate('/about'); // Direct to the About page after sign-in
        }*/
        
            async function handleCallbackResponse(response) {
                console.log("Encoded JWT ID token: " + response.credential);
                const userObject = jwtDecode(response.credential);  // Decode JWT token
                console.log("User object:", userObject);
    
                const { name, email, sub: googleId } = userObject;
    
                // Check if user exists and save to database if new
                try {
                    const res = await axios.post('/api/users', { name, email, googleId });
                    console.log('User data:', res.data);
                    handleSignIn(); // Notify Main.js about successful sign-in
                    navigate('/about'); // Direct to the About page after sign-in
                } catch (error) {
                    console.error('Error saving user data:', error);
                }
            }
        }, [navigate, handleSignIn]);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    
    return (
        <div className="about-page">
            
            <section className="content">
                <div className="text-content">
                    <h3>What is ScanEye</h3>
                    <p>
                        A web-based application to detect diabetic retinopathy using retinal images of the eye through ML/AI algorithms.
                    </p>
                    <p>
                        Our purpose is to provide a reliable tool for early detection and monitoring of diabetic retinopathy, a common complication of diabetes that affects the eyes. We aim to facilitate timely diagnosis and improve patient outcomes.
                    </p>
                    <button className="contact-button" onClick={() => navigate('/contact')}>
                        Contact Us
                    </button>
                    <div className="sign-in-container">
                    
                    <div id="signInDiv"></div> {/* Sign-In button will be rendered here */}
            </div>
                </div>
                
                <img className="retina-image" src={retinaImage} alt="Retina" />
                
                
            </section>
        </div>
    );
};

export default AboutPage;

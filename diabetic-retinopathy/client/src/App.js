import React, { useEffect,useState } from 'react';
import Upload from './components/upload';
import { useNavigate } from 'react-router-dom';
import logoimage from './assests/logo1.png'; // Corrected path


import './App.css';

function App() {
  // eslint-disable-next-line no-unused-vars
  const navigate = useNavigate();
  const [backendData,setBackendData] = useState([{}])


  function handleCallbackResponse(response) {
    console.log("Encoded JWT ID token: " + response.credential);
    window.location.href = "/upload";
    navigate('/about');
  }

  useEffect(() => {
    /* global google */
    fetch("/api")
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));


  
    
    google.accounts.id.initialize({
      client_id: "41122690496-56u8g711vqh0h5afagp1lts5bch9r8hk.apps.googleusercontent.com",
      callback: handleCallbackResponse
    });
    //google.accounts.id.renderButton(
     // document.getElementById("signInDiv"),
      //{ theme: "filled_white", size: "large", prompt: "select_account" }
    //);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [navigate]);

  return (
    <div className="App">
      <div id="signInDiv"></div>
      <div className="title-container">
        <h2>ScanEy<img className="logoim" src={logoimage} alt="Logo Image" /></h2>
        <h6>Diabetic Retinopathy Detection System</h6>
      </div>
    </div>
  );
}

export default App;

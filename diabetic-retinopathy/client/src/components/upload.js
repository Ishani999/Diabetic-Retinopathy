import React, { useState } from 'react';
import axios from 'axios';
import './Upload.css';
import { useNavigate } from 'react-router-dom';



const Upload = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [imageUrl, setImageUrl] = useState('');
    const [prediction, setPrediction] = useState(null);
    //const [annotatedImageUrl, setAnnotatedImageUrl] = useState('');
    
    const [loading, setloading] = useState(false);
    const navigate = useNavigate();
    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setImageUrl(URL.createObjectURL(event.target.files[0]));
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            console.error('No file selected for upload.');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);
        setloading(true);
        
        try {

          const token = localStorage.getItem('token'); // Retrieve the token from localStorage
          console.log('Token:', token);

          const response = await axios.post('/uploads', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
                'Authorization': 'Bearer ' + token
            }
        });

        setPrediction(response.data.prediction);
      } catch (error) {
          console.error('Error uploading file:', error);
      } finally {
        setloading(false); // Reset loading state
    }
  };
  const renderDescription = () => {
    if (prediction === 'Mild') {
        return (
            <p>
                Mild diabetic retinopathy shows early signs of damage to the retina, which may not require immediate treatment but requires regular monitoring to prevent progression.
            </p>
        );
    } else if (prediction === 'Moderate') {
        return (
            <p>
                Moderate diabetic retinopathy indicates a more advanced stage with noticeable damage to blood vessels in the retina, requiring closer monitoring and possible treatment.
            </p>
        );
    } else if (prediction === 'Severe') {
        return (
            <p>
                Severe diabetic retinopathy signifies extensive damage to the retina's blood vessels, increasing the risk of vision loss without prompt medical intervention.
            </p>
        );
    } else if (prediction === 'No_DR') {
      return (
          <p>
               The retina appears normal without any detectable damage to the blood vessels.There are no signs of diabetic retinopathy in the eye
          </p>
      );
    } else if (prediction === 'Proliferate_DR') {
      return (
          <p>
            "Proliferate DR" is an advanced stage of diabetic retinopathy characterized by the growth of abnormal blood vessels in the retina. These blood vessels are fragile and prone to bleeding, leading to severe vision problems and potentially blindness if left untreated. Prompt medical intervention, such as laser treatment or surgery, is often necessary to prevent further vision loss.
          </p>
    );
}
    

    return null; // Handle cases where prediction is not set or recognized
};
    const navigateToHistory = () => {
      navigate('/history');
    };

  return (
    <div className="upload-container">
        <div className="upload-section">
            <h2>Please Upload Eye Image Here...</h2>
            <form onSubmit={(e) => { e.preventDefault(); handleUpload(); }}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit">Upload</button>
            </form>
            {imageUrl && (
                    <div className="uploaded-image">
                        <img src={imageUrl} alt="Uploaded" />
                    </div>
            )}
        </div>
        <div className="analysis-section">
            {prediction && (
                <div className="analysis-report">
                    <h2>Analysis Report</h2>
                    <h5>Prediction: {prediction}</h5>
                    {renderDescription()}
                    
                </div>
            )}
          
        <div className="history-button-container">
            <button onClick={navigateToHistory}>View History</button>
        </div>
      </div>
    </div>
    
  );
};

export default Upload;
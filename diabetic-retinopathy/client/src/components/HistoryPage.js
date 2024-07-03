import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './HistoryPage.css';

const HistoryPage = () => {
  const [uploadHistory, setUploadHistory] = useState([]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await axios.get('http://localhost:3000/api/history');
        console.log('History data:', response.data); // Log response data for debugging
        setUploadHistory(response.data);
      } catch (error) {
        console.error('Error fetching upload history:', error);
      }
    };

    fetchHistory();
  }, []);
  const clearHistory = async () => {
    try {
        await axios.delete('http://localhost:3000/api/history');
        setUploadHistory([]); // Clear local state
        alert('Upload history cleared successfully');
    } catch (error) {
        console.error('Error clearing upload history:', error);
        alert('Failed to clear upload history');
    }
};
  return (
    <div className="history-page">
      <h2>History</h2>
      <button onClick={clearHistory} className="clear-history-btn">Clear History</button>
      <table>
        <thead>
          <tr>
            <th>Upload Date and Time</th>
            <th>Prediction Result</th>
          </tr>
        </thead>
        <tbody>
        {uploadHistory.length > 0 ? (
          uploadHistory.map((upload) => (
            <tr key={upload.id}>
              <td>{new Date(upload.uploaded_at).toLocaleString()}</td>
              <td>{upload.prediction_result}</td>
            </tr>
          ))
        ) : (
          <tr>
            <td colSpan="2">No history available</td>
          </tr>
        )}
      </tbody>
    </table>
  </div>
);
};

export default HistoryPage;

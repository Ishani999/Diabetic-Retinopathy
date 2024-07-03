import React from 'react';
import { useLocation } from 'react-router-dom';

const Dashboard = () => {
    const location = useLocation();
    const { prediction, annotatedImageUrl } = location.state;

    return (
        <div className="dashboard-container">
            <h2>Dashboard Interface</h2>
            {prediction && <p>Prediction: {prediction}</p>}
            {annotatedImageUrl && <img src={annotatedImageUrl} alt="Annotated" />}
        </div>
    );
};

export default Dashboard;

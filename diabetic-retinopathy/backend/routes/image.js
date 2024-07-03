const express = require('express');
const router = express.Router();
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');


router.post('/upload', async (req, res) => {
  try {
    if (!req.files || !req.files.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Assuming file is passed in form-data as 'file'
    const file = req.files.file;
    const fileName = file.name;
    const filePath = path.join(__dirname, '../uploads', fileName);

    // Save file to server
    file.mv(filePath, async (err) => {
      if (err) {
        console.error('Error saving file:', err);
        return res.status(500).json({ error: 'Failed to upload file' });
      }

      try {
        // Prepare form data to send to Flask
        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath), { filename: fileName });

        // Send POST request to Flask API
        const flaskResponse = await axios.post('http://localhost:5001/upload', formData, {
          headers: {
            ...formData.getHeaders(),
          },
        });

        // Handle Flask API response
        return res.json(flaskResponse.data);
      } catch (error) {
        console.error('Error communicating with Flask:', error);
        return res.status(500).json({ error: 'Internal server error' });
      } finally {
        // Clean up: delete uploaded file from server
        fs.unlinkSync(filePath);
      }
    });
  } catch (error) {
    console.error('Error uploading file:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;

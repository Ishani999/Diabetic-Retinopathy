require('dotenv').config(); 
const express = require('express');
const bodyParser = require('body-parser');
const fileUpload = require('express-fileupload');
const axios = require('axios');
const path = require('path');
const FormData = require('form-data');
const db = require('./config/db'); 
const admin = require('firebase-admin');
const { OAuth2Client } = require('google-auth-library');
const multer = require('multer');

const jwt = require('jsonwebtoken');
const upload = multer();
const app = express();
const port = process.env.PORT || 5000; // Adjust port as needed
const CLIENT_ID = process.env.CLIENT_ID;
const CLIENT_SECRET = process.env.CLIENT_SECRET;


// Initialize Firebase Admin SDK
const serviceAccount = require('./serviceAccountKey.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  storageBucket: 'effectivesolutions.appspot.com' // Replace with your Firebase Storage bucket URL
});

const bucket = admin.storage().bucket();
const client = new OAuth2Client(CLIENT_ID);


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(fileUpload());

app.use(express.static(path.join(__dirname, '..', 'client', 'build')));


// Route to handle saving user data from Google OAuth
app.post('/api/users', (req, res) => {
  const { name, email, googleId } = req.body;

  // Check if user already exists in the database based on googleId
  const selectQuery = 'SELECT * FROM users WHERE google_id = ?';
  db.query(selectQuery, [googleId], (err, results) => {
      if (err) {
          console.error('Error checking user existence:', err);
          return res.status(500).json({ error: 'Internal server error' });
      }

      if (results.length > 0) {
          // User already exists, send back existing user data
          console.log('User already exists:', results[0]);
          res.json(results[0]);
      } else {
          // User doesn't exist, insert new user into the database
          const insertQuery = 'INSERT INTO users (name, email, google_id, created_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)';
          db.query(insertQuery, [name, email, googleId], (err, result) => {
              if (err) {
                  console.error('Error inserting user data:', err);
                  return res.status(500).json({ error: 'Error inserting user data' });
              }

              // Retrieve inserted user data
              const selectInsertedUserQuery = 'SELECT * FROM users WHERE id = ?';
              db.query(selectInsertedUserQuery, [result.insertId], (err, insertedUser) => {
                  if (err) {
                      console.error('Error fetching inserted user:', err);
                      return res.status(500).json({ error: 'Internal server error' });
                  }

                  console.log('Inserted user:', insertedUser[0]);
                  res.json(insertedUser[0]);
              });
          });
      }
  });
});

  
// Route to serve uploaded images from Flask
app.get('/api/uploads/:filename', (req, res) => {
  const { filename } = req.params;
  const filePath = path.join(__dirname, `ml_model/static/uploads/${filename}`);
  console.log(`Serving file: ${filePath}`);
  res.sendFile(filePath);
});

app.get('/api/uploads/:filename', (req, res) => {
  const { filename } = req.params;
  const firebaseUrl = `https://storage.googleapis.com/${bucket.name}/${filename}`;
  console.log(`Serving file: ${firebaseUrl}`);
  res.redirect(firebaseUrl);
});

/*const verifyToken = (req, res, next) => {
  if (!req.headers.authorization || !req.headers.authorization.startsWith('Bearer ')) {
    return res.status(401).send('Unauthorized');
  }

  const token = req.headers.authorization.split('Bearer ')[1];
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.userName = decoded.name; // Assuming name is available in the token
    req.userEmail = decoded.email; // Assuming email is the unique identifier
    next();
  } catch (error) {
    console.error('Error decoding token:', error);
    return res.status(401).send('Unauthorized');
  }
};
*/
// Route to handle image upload and prediction
app.post('/uploads', async (req, res) => {
  const { googleId } = req.body;
  console.log(`POST request received at /upload`);

  const { name: filename, data, mimetype } = req.files.file;
  console.log(`Received file: ${filename}`);

  // Fetch user ID based on googleId
  /*const selectQuery = 'SELECT id FROM users WHERE google_id = ?';
  db.query(selectQuery, [googleId], async (err, results) => {
      if (err) {
          console.error('Error fetching user data:', err);
          return res.status(500).json({ error: 'Internal server error' });
      }

      if (results.length === 0) {
          return res.status(404).json({ error: 'User not found' });
      }

      const userId = 1;
      console.log('User ID:', userId);

      // Upload file to Firebase Storage
   });*/
  const blob = bucket.file(filename);
  const blobStream = blob.createWriteStream({
      metadata: {
          contentType: mimetype,
      },
  });

  blobStream.on('error', (err) => {
      console.error('Error uploading to Firebase Storage:', err);
      res.status(500).send('Error uploading to Firebase Storage');
  });

  blobStream.on('finish', async () => {
          // File upload to Firebase Storage is complete
      const firebaseUrl = `https://storage.googleapis.com/${bucket.name}/${blob.name}`;
      console.log('File uploaded to Firebase Storage:', firebaseUrl);

      // Forward file upload request to Flask for processing
      const formData = new FormData();
      formData.append('file', data, filename);

      console.log('Sending file to Flask:', filename);

      try {
              // Forward file upload request to Flask
          const flaskResponse = await axios.post('http://localhost:5001/upload', formData, {
              headers: {
                ...formData.getHeaders(),
              },
          });
          console.log('Flask response:', flaskResponse.data);
          const prediction = flaskResponse.data.prediction;
          
          // Insert data into usage_history table
          const UserId = 1;
          const insertQuery = 'INSERT INTO usage_history (user_id, filename, uploaded_at, prediction_result) VALUES (?, ?, NOW(), ?)';
          db.query(insertQuery, [UserId, filename, prediction], (err, result) => {
              if (err) {
                  console.error('Error inserting data into database:', err);
                   return res.status(500).send('Error inserting data into database');
              }

              console.log('Data inserted into database:', result);
              res.status(200).send(flaskResponse.data);
          });
              
      } catch (error) {
          console.error('Error forwarding file to Flask:', error);
          res.status(500).send('Error forwarding file to Flask');
      }
  });

  // End the blobStream with file data
  blobStream.end(data);
  });

app.get('/api/history', (req, res) => {
  const sql = 'SELECT filename, uploaded_at, prediction_result FROM usage_history';
  db.query(sql, (err, results) => {
    if (err) {
      console.error('Error fetching history data:', err);
      return res.status(500).json({ error: 'Internal server error' });
    }
    console.log('History data fetched from database:', results);
    res.json(results);
  });
});
//Clear History
const authenticateMiddleware = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1]; // Assuming token is sent in Authorization header

  if (!token) {
      return res.status(401).json({ message: 'Unauthorized' });
  }

  try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      req.user = decoded; // Attach decoded user information to req.user
      next();
  } catch (error) {
      console.error('Error verifying token:', error);
      return res.status(403).json({ message: 'Invalid token' });
  }
};


app.delete('/api/history', async (req, res) => {
  try {
      // Assuming userId is extracted from authentication middleware
      //const userId = req.user.id;

      // Perform database operation to delete history for userId
      db.query('DELETE * FROM usage_history');

      res.status(200).json({ message: 'Upload history cleared successfully' });
  } catch (error) {
      console.error('Error clearing upload history:', error);
      res.status(500).json({ error: 'Failed to clear upload history' });
  }
});



// Route to render the React frontend (if applicable)
app.use(express.static(path.join(__dirname, '..', 'client', 'build')));
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'client', 'build', 'index.html'));
});
db.connect((err) => {
    if (err) {
      console.error('Error connecting to MySQL:', err);
      return;
    }
    console.log('Connected to MySQL database.');
});  
// Start server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
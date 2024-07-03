// routes/auth.js
const express = require('express');
const router = express.Router();
const db = require('../config/db');

// Example route for user login
router.post('/login', (req, res) => {
  const { username, password } = req.body;

  db.query('SELECT * FROM users WHERE username = ? AND password = ?', [username, password], (err, results) => {
    if (err) {
      console.error('Error querying users:', err);
      res.status(500).json({ error: 'Internal Server Error' });
      return;
    }

    if (results.length > 0) {
      // User found, authenticate
      res.json({ message: 'Login successful', user: results[0] });
    } else {
      // User not found or credentials incorrect
      res.status(401).json({ error: 'Invalid credentials' });
    }
  });
});

// Add more routes for registration, user management, etc.

module.exports = router;

const express = require('express');
const router = express.Router();
const authRoutes = require('./auth');
const imageRoutes = require('./image');

// Mounting individual route modules
router.use('/auth', authRoutes);
router.use('/image', imageRoutes);

module.exports = router;

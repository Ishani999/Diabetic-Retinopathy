// Example User model (User.js)
const db = require('../config/db');

const User = {
  getAll: (callback) => {
    db.query('SELECT * FROM users', callback);
  },
  getById: (id, callback) => {
    db.query('SELECT * FROM users WHERE id = ?', [id], callback);
  },
  create: (newUser, callback) => {
    db.query('INSERT INTO users SET ?', newUser, callback);
  },
  updateById: (id, user, callback) => {
    db.query('UPDATE users SET ? WHERE id = ?', [user, id], callback);
  },
  deleteById: (id, callback) => {
    db.query('DELETE FROM users WHERE id = ?', [id], callback);
  }
};

module.exports = User;

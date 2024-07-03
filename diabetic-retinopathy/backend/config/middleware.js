
const authMiddleware = (req, res, next) => {
    
    console.log('Auth middleware executed');
    next();
  };
  
  module.exports = {
    authMiddleware
  };
  
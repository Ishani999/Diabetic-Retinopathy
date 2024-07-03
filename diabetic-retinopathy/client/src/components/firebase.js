
import { initializeApp } from 'firebase/app';
import { getStorage } from 'firebase/storage';

const firebaseConfig = {
apiKey: "AIzaSyC2dc7q5o1Qb9p3Kp8oNryKz8GfbkuPdhI",
  authDomain: "effectivesolutions.firebaseapp.com",
  projectId: "effectivesolutions",
  storageBucket: "effectivesolutions.appspot.com",
  messagingSenderId: "41122690496",
  appId: "1:41122690496:web:380998dae3691b9f3e0425",
  measurementId: "G-V3SMB2B7B5"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app); // Initialize storage correctly

export { storage, app as firebase }; // Export storage and app as firebase
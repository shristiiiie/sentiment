import React, { useState } from 'react';
import '../CSS/LoginPage.css';
import { useNavigate } from 'react-router-dom';

const LoginPage = () => {
  const [isAdmin, setIsAdmin] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    setTimeout(() => {
      if (isAdmin) {
        if (email === 'admin@test.com' && password === 'admin123') {
          alert('Welcome, Admin!');
          localStorage.setItem("userRole", "admin"); // ✅ Store role
          navigate('/admin-dashboard');
        } else {
          setError('Invalid admin credentials');
        }
      } else {
        if (email === 'student@test.com' && password === 'student123') {
          alert('Welcome, Student!');
          localStorage.setItem("userRole", "student"); // ✅ Store role
          navigate('/student-dashboard');
        } else {
          setError('Invalid student credentials');
        }
      }
      setLoading(false);
    }, 1000);
  };

  const switchToAdmin = () => {
    setIsAdmin(true);
    setEmail('');
    setPassword('');
    setError('');
  };

  return (
    <div className={`container ${isAdmin ? 'admin-theme' : 'student-theme'}`}>
      <div className="left-panel">
        <div className="title-text">
          {isAdmin ? '' : ''}
        </div>
      </div>
      <div className="login-box">
        <div className="info">
          <h1 className="dashboard-heading">
      {isAdmin ? 'Admin Dashboard' : 'Student Dashboard'}
    </h1>
          <p className="subheading">
            {isAdmin
              ? 'Access the admin dashboard to monitor feedback and analytics'
              : 'You are connecting to Student Feedback Form'}
          </p>
          <h2 className="signin-text">SIGN IN</h2>
        </div>

        <form onSubmit={handleLogin}>
          <input
            type="text"
            placeholder="Email Address"
            value={email}
            className={error ? 'input error' : 'input'}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            className={error ? 'input error' : 'input'}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          {error && <p className="error-text">{error}</p>}
          <button type="submit" className="button" disabled={loading}>
            {loading ? <span className="spinner"></span> : 'Sign in'}
          </button>
        </form>

        {!isAdmin && (
          <div className="admin-link">
            OR<br />If you’re ADMIN, please{' '}
            <span onClick={switchToAdmin} className="click-here">click here</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default LoginPage;

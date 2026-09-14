import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/common/Navbar';
import HomePage from './pages/HomePage';
import RecordedPage from './pages/RecordedPage';
import RecordedXAIPage from './pages/RecordedXAIPage';
import LivePage from './pages/LivePage';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/recorded" element={<RecordedPage />} />
        <Route path="/recorded/xai" element={<RecordedXAIPage />} />
        <Route path="/live" element={<LivePage />} />
      </Routes>
    </BrowserRouter>
  );
}

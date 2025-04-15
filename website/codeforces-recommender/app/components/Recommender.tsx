'use client';

import { useState } from 'react';
import axios from 'axios';

export default function Recommender() {
  const [handle, setHandle] = useState('');
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  interface RecommendationResponse {
    recommendations: string[];
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    setError('');
    setRecommendations([]);
    setLoading(true);

    try {
      const response = await axios.post<RecommendationResponse>('https://your-app.onrender.com/recommend', { handle });
      setRecommendations(response.data.recommendations);
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.error || 'Something went wrong. Please try again.');
      } else {
        setError('Something went wrong. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md bg-white p-6 rounded-lg shadow-md">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="handle" className="block text-sm font-medium text-gray-700">
            Codeforces Handle
          </label>
          <input
            id="handle"
            type="text"
            value={handle}
            onChange={(e) => setHandle(e.target.value)}
            placeholder="Enter your handle (e.g., Hamim99)"
            className="mt-1 w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className={`w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            loading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
      </form>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {recommendations.length > 0 && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold text-gray-800">Recommended Problems</h2>
          <ul className="mt-2 space-y-2">
            {recommendations.map((rec, index) => (
              <li key={index} className="p-3 bg-gray-100 rounded-md">
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
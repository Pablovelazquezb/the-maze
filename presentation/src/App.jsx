import { useState, useEffect } from 'react'
import MazeVisualizer from './components/MazeVisualizer'
import './index.css'

function App() {
  const [currentSlide, setCurrentSlide] = useState(0)

  const slides = [
    // Slide 0: Title
    <div className="flex flex-col items-center justify-center h-full text-center space-y-8 animate-fade-in">
      <div className="inline-block p-4 rounded-2xl bg-slate-800/50 border border-slate-700 backdrop-blur-sm mb-4">
        <div className="text-emerald-400 font-mono text-sm tracking-widest uppercase">COSC 4368 — Group 14</div>
      </div>
      <h1 className="text-6xl md:text-8xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-emerald-400 drop-shadow-sm pb-4">
        The Silent Cartographer
      </h1>
      <p className="text-2xl text-slate-400 max-w-3xl font-light">
        Solving dynamic hazard mazes using Hybrid Q-Learning
      </p>
      <div className="pt-12">
        <p className="text-slate-500 font-mono text-sm">Use arrow keys to navigate</p>
      </div>
    </div>,

    // Slide 1: The Problem
    <div className="flex flex-col h-full justify-center max-w-5xl mx-auto space-y-10 animate-fade-in">
      <h2 className="text-5xl font-bold text-white border-b border-slate-800 pb-6">The Challenge</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-slate-800/40 p-8 rounded-3xl border border-slate-700/50 hover:bg-slate-800/60 transition-colors">
          <div className="text-4xl mb-4">🗺️</div>
          <h3 className="text-2xl font-bold text-slate-200 mb-3">Scale</h3>
          <p className="text-slate-400 text-lg leading-relaxed">
            Navigating a massive 64x64 grid (4,096 cells) starting with zero knowledge. Sparse rewards make standard Reinforcement Learning computationally infeasible.
          </p>
        </div>
        <div className="bg-slate-800/40 p-8 rounded-3xl border border-slate-700/50 hover:bg-slate-800/60 transition-colors">
          <div className="text-4xl mb-4">🔥</div>
          <h3 className="text-2xl font-bold text-slate-200 mb-3">Dynamic Hazards</h3>
          <p className="text-slate-400 text-lg leading-relaxed">
            Death pits rotate 90° every 5 actions. Static pathfinding algorithms (A*, DFS) fail because paths that are initially safe become lethal during execution.
          </p>
        </div>
      </div>
    </div>,

    // Slide 2: The Solution
    <div className="flex flex-col h-full justify-center max-w-5xl mx-auto space-y-8 animate-fade-in">
      <h2 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-400 pb-2">Hybrid Q-Learning</h2>
      <p className="text-xl text-slate-300">Combining the speed of classical search with the adaptability of RL.</p>
      
      <div className="space-y-6 pt-6">
        <div className="flex items-start space-x-6 bg-slate-800/30 p-6 rounded-2xl border border-slate-700/50">
          <div className="bg-blue-500/20 text-blue-400 p-4 rounded-xl font-bold text-2xl">1</div>
          <div>
            <h3 className="text-2xl font-bold text-white mb-2">Model-Based Exploration</h3>
            <p className="text-slate-400 text-lg">Uses BFS to systematically route to the nearest unexplored frontier until the goal is found, drastically reducing exploration time.</p>
          </div>
        </div>
        
        <div className="flex items-start space-x-6 bg-slate-800/30 p-6 rounded-2xl border border-slate-700/50">
          <div className="bg-purple-500/20 text-purple-400 p-4 rounded-xl font-bold text-2xl">2</div>
          <div>
            <h3 className="text-2xl font-bold text-white mb-2">Model-Free Exploitation</h3>
            <p className="text-slate-400 text-lg">Computes optimal BFS path and seeds the Q-table. If a rotating hazard blocks the planned path, it dynamically falls back to Q-Learning to maneuver around the danger.</p>
          </div>
        </div>
      </div>
    </div>,

    // Slide 3: Live Demo
    <div className="flex flex-col h-full items-center justify-center animate-fade-in w-full">
      <div className="w-full max-w-6xl flex justify-between items-end mb-6 px-4">
        <div>
          <h2 className="text-4xl font-bold text-white">Live Execution</h2>
          <p className="text-slate-400">Zero-Shot Test on Maze Beta (Hazards enabled)</p>
        </div>
        <div className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-4 py-2 rounded-lg font-mono text-sm">
          Status: Agent Loaded
        </div>
      </div>
      <MazeVisualizer />
    </div>,

    // Slide 4: Results
    <div className="flex flex-col h-full justify-center max-w-5xl mx-auto space-y-10 animate-fade-in">
      <h2 className="text-5xl font-bold text-white text-center mb-8">Performance Metrics</h2>
      
      <div className="overflow-hidden rounded-3xl border border-slate-700 shadow-2xl bg-slate-800/50 backdrop-blur-sm">
        <table className="w-full text-left">
          <thead>
            <tr className="bg-slate-900/80">
              <th className="p-6 text-slate-300 font-semibold text-lg border-b border-slate-700">Metric</th>
              <th className="p-6 text-slate-300 font-semibold text-lg border-b border-slate-700">Maze Alpha (Train)</th>
              <th className="p-6 text-emerald-400 font-bold text-lg border-b border-slate-700 bg-emerald-950/20">Maze Beta (Zero-Shot Test)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50 text-slate-300 text-lg">
            <tr className="hover:bg-slate-800/50 transition-colors">
              <td className="p-6 font-medium">Success Rate</td>
              <td className="p-6">92.5%</td>
              <td className="p-6 text-emerald-400 font-bold bg-emerald-950/10">100.0%</td>
            </tr>
            <tr className="hover:bg-slate-800/50 transition-colors">
              <td className="p-6 font-medium">Avg Turns to Solution</td>
              <td className="p-6">8080.2</td>
              <td className="p-6 text-emerald-400 font-bold bg-emerald-950/10">251.2</td>
            </tr>
            <tr className="hover:bg-slate-800/50 transition-colors">
              <td className="p-6 font-medium">Death Rate</td>
              <td className="p-6">0.0001</td>
              <td className="p-6 text-emerald-400 font-bold bg-emerald-950/10">0.0000</td>
            </tr>
            <tr className="hover:bg-slate-800/50 transition-colors">
              <td className="p-6 font-medium">Exploration Efficiency</td>
              <td className="p-6">0.106</td>
              <td className="p-6 text-emerald-400 font-bold bg-emerald-950/10">0.903</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="text-center text-slate-400 text-xl pt-4">
        The model successfully generalized avoiding all dynamic hazards in a new environment without retraining!
      </p>
    </div>
  ]

  const nextSlide = () => setCurrentSlide((s) => Math.min(slides.length - 1, s + 1))
  const prevSlide = () => setCurrentSlide((s) => Math.max(0, s - 1))

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ') nextSlide()
      if (e.key === 'ArrowLeft') prevSlide()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  return (
    <div className="h-screen w-screen bg-[#0a0f18] text-slate-200 overflow-hidden font-sans selection:bg-purple-500/30">
      
      {/* Background ambient effects */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-purple-600/10 blur-[100px] rounded-full"></div>
        <div className="absolute bottom-0 right-0 w-[40rem] h-[40rem] bg-emerald-600/5 blur-[120px] rounded-full"></div>
      </div>

      {/* Main Slide Content */}
      <div className="relative z-10 w-full h-full p-8 md:p-16">
        {slides[currentSlide]}
      </div>

      {/* Navigation Footer */}
      <div className="absolute bottom-8 left-0 w-full flex justify-between items-center px-12 z-20">
        <div className="flex space-x-2">
          {slides.map((_, i) => (
            <div 
              key={i} 
              className={`h-1.5 rounded-full transition-all duration-500 ${i === currentSlide ? 'w-8 bg-purple-500' : 'w-2 bg-slate-700'}`}
            />
          ))}
        </div>
        <div className="flex space-x-4">
          <button 
            onClick={prevSlide}
            disabled={currentSlide === 0}
            className="p-3 rounded-full bg-slate-800/50 border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-700 disabled:opacity-30 transition-all backdrop-blur-md"
          >
            ←
          </button>
          <button 
            onClick={nextSlide}
            disabled={currentSlide === slides.length - 1}
            className="p-3 rounded-full bg-slate-800/50 border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-700 disabled:opacity-30 transition-all backdrop-blur-md"
          >
            →
          </button>
        </div>
      </div>
    </div>
  )
}

export default App

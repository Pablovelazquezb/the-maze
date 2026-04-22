import React, { useState, useEffect, useRef } from 'react';
import demoData from '../assets/demo_data.json';

const MazeVisualizer = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(50); // ms per step
  
  const { walls, pits, teleporters, confusion_pads, start_pos, goal_pos, path } = demoData;
  const maxSteps = path.length - 1;

  // Use a ref for auto-play interval
  const timerRef = useRef(null);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= maxSteps) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, speed);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, maxSteps, speed]);

  const togglePlay = () => {
    if (!isPlaying && currentStep >= maxSteps) {
      setCurrentStep(0);
      setIsPlaying(true);
    } else {
      setIsPlaying(!isPlaying);
    }
  };
  
  const reset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };
  
  // Since rendering a 64x64 grid of DOM nodes is slow, we use Canvas
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Canvas dimensions
    const width = canvas.width;
    const height = canvas.height;
    
    // Grid sizing
    const cols = 64;
    const rows = 64;
    const cellW = width / cols;
    const cellH = height / rows;

    // Clear canvas
    ctx.fillStyle = '#0f172a'; // slate-900 background
    ctx.fillRect(0, 0, width, height);

    // Draw walls
    ctx.fillStyle = '#334155'; // slate-700
    walls.forEach(([x, y]) => {
      ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
    });

    // Draw goal
    ctx.fillStyle = '#10b981'; // emerald-500
    ctx.fillRect(goal_pos[0] * cellW, goal_pos[1] * cellH, cellW, cellH);

    // Draw pits (Fire)
    ctx.fillStyle = '#ef4444'; // red-500
    pits.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x * cellW + cellW/2, y * cellH + cellH/2, cellW/2 * 0.8, 0, 2*Math.PI);
      ctx.fill();
    });

    // Draw Teleporters
    ctx.fillStyle = '#3b82f6'; // blue-500
    teleporters.forEach(([x, y]) => {
      ctx.fillRect(x * cellW + 1, y * cellH + 1, cellW - 2, cellH - 2);
    });

    // Draw Confusion
    ctx.fillStyle = '#eab308'; // yellow-500
    confusion_pads.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.moveTo(x * cellW + cellW/2, y * cellH);
      ctx.lineTo(x * cellW + cellW, y * cellH + cellH);
      ctx.lineTo(x * cellW, y * cellH + cellH);
      ctx.fill();
    });

    // Draw Path Trail
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)'; // purple-500 with opacity
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(path[0][0] * cellW + cellW/2, path[0][1] * cellH + cellH/2);
    for (let i = 1; i <= currentStep; i++) {
      ctx.lineTo(path[i][0] * cellW + cellW/2, path[i][1] * cellH + cellH/2);
    }
    ctx.stroke();

    // Draw Bot
    const currentPos = path[currentStep];
    ctx.fillStyle = '#a855f7'; // purple-500
    ctx.shadowColor = '#d8b4fe';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(currentPos[0] * cellW + cellW/2, currentPos[1] * cellH + cellH/2, cellW/2, 0, 2*Math.PI);
    ctx.fill();
    ctx.shadowBlur = 0; // reset

  }, [currentStep, walls, pits, teleporters, confusion_pads, goal_pos, path]);

  return (
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto space-y-6">
      
      <div className="relative rounded-xl overflow-hidden border border-slate-700 shadow-2xl bg-slate-900 p-2">
        <canvas 
          ref={canvasRef} 
          width={640} 
          height={640} 
          className="w-full max-w-[60vh] aspect-square rounded bg-slate-900"
        />
        
        {currentStep === maxSteps && demoData.success && (
          <div className="absolute inset-0 bg-emerald-500/20 backdrop-blur-sm flex items-center justify-center">
            <div className="bg-slate-900 border border-emerald-500 text-emerald-400 px-8 py-4 rounded-xl text-3xl font-bold shadow-2xl animate-pulse">
              SUCCESS! Goal Reached
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-col w-full bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 backdrop-blur-sm">
        <div className="flex justify-between items-center mb-4 text-slate-300">
          <div className="font-mono text-sm">
            Step: <span className="text-purple-400 font-bold">{currentStep}</span> / {maxSteps}
          </div>
          <div className="flex space-x-4">
            <button 
              onClick={() => setSpeed(100)} 
              className={`px-3 py-1 rounded text-xs font-bold transition-colors ${speed === 100 ? 'bg-purple-600 text-white' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'}`}
            >
              1x
            </button>
            <button 
              onClick={() => setSpeed(20)} 
              className={`px-3 py-1 rounded text-xs font-bold transition-colors ${speed === 20 ? 'bg-purple-600 text-white' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'}`}
            >
              5x
            </button>
            <button 
              onClick={() => setSpeed(5)} 
              className={`px-3 py-1 rounded text-xs font-bold transition-colors ${speed === 5 ? 'bg-purple-600 text-white' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'}`}
            >
              MAX
            </button>
          </div>
        </div>
        
        <input 
          type="range" 
          min="0" 
          max={maxSteps} 
          value={currentStep} 
          onChange={(e) => {
            setCurrentStep(parseInt(e.target.value));
            setIsPlaying(false);
          }}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500 mb-6"
        />

        <div className="flex justify-center space-x-4">
          <button 
            onClick={reset}
            className="px-6 py-3 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 font-semibold transition-all shadow-lg active:scale-95"
          >
            Reset
          </button>
          <button 
            onClick={togglePlay}
            className={`px-8 py-3 rounded-lg font-bold transition-all shadow-lg active:scale-95 w-40 ${
              isPlaying 
                ? 'bg-amber-500 hover:bg-amber-400 text-amber-950 shadow-amber-500/20' 
                : 'bg-emerald-500 hover:bg-emerald-400 text-emerald-950 shadow-emerald-500/20'
            }`}
          >
            {isPlaying ? 'Pause' : 'Play Simulation'}
          </button>
        </div>
      </div>
      
      <div className="flex space-x-6 text-sm text-slate-400">
        <div className="flex items-center"><span className="w-3 h-3 rounded-full bg-red-500 mr-2 shadow-[0_0_8px_rgba(239,68,68,0.6)]"></span> Death Pit</div>
        <div className="flex items-center"><span className="w-3 h-3 rounded-sm bg-blue-500 mr-2"></span> Teleporter</div>
        <div className="flex items-center"><span className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-b-[10px] border-b-yellow-500 mr-2"></span> Confusion Pad</div>
      </div>

    </div>
  );
};

export default MazeVisualizer;

import { useState, useEffect } from 'react';
import { Card, CardBody, Progress } from '@heroui/react';
import { motion } from 'framer-motion';
import {
  Brain,
  Link2,
  Zap,
  Coins,
  Clock,
  BookOpen,
  TrendingUp,
  Activity,
} from 'lucide-react';
import { useNetworkStore } from '../../stores/networkStore';

// Simulated brain integration state (would come from real WASM/relay in production)
interface BrainState {
  connectionHealth: 'connected' | 'degraded' | 'disconnected';
  relayLatency: number;
  operationsToday: number;
  ruvEarnedToday: number;
  halvingEpoch: number;
  epochProgress: number;
  topCategories: { name: string; queries: number; color: string }[];
  brainUptime: number;
  totalKnowledgeEntries: number;
}

function useBrainState(): BrainState {
  const { isRelayConnected, stats, credits } = useNetworkStore();
  const [brainState, setBrainState] = useState<BrainState>({
    connectionHealth: 'disconnected',
    relayLatency: 0,
    operationsToday: 0,
    ruvEarnedToday: 0,
    halvingEpoch: 1,
    epochProgress: 0.37,
    topCategories: [
      { name: 'Code Patterns', queries: 1842, color: 'sky' },
      { name: 'Architecture', queries: 1203, color: 'violet' },
      { name: 'Security', queries: 891, color: 'amber' },
      { name: 'Performance', queries: 654, color: 'emerald' },
      { name: 'DevOps', queries: 412, color: 'cyan' },
    ],
    brainUptime: 0,
    totalKnowledgeEntries: 0,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setBrainState((prev) => {
        const health = isRelayConnected
          ? stats.latency < 100
            ? 'connected'
            : 'degraded'
          : 'disconnected';

        return {
          ...prev,
          connectionHealth: health,
          relayLatency: stats.latency,
          operationsToday: Math.floor(stats.tasksCompleted * 2.3),
          ruvEarnedToday: Math.round(credits.earned * 0.4 * 100) / 100,
          brainUptime: prev.brainUptime + 1,
          totalKnowledgeEntries: 24_831 + Math.floor(prev.brainUptime / 5),
          topCategories: prev.topCategories.map((cat) => ({
            ...cat,
            queries: cat.queries + Math.floor(Math.random() * 3),
          })),
        };
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [isRelayConnected, stats.latency, stats.tasksCompleted, credits.earned]);

  return brainState;
}

const healthConfig = {
  connected: {
    label: 'Healthy',
    color: 'emerald',
    bgClass: 'bg-emerald-500/10 border-emerald-500/30',
    textClass: 'text-emerald-400',
    dotClass: 'bg-emerald-400',
  },
  degraded: {
    label: 'Degraded',
    color: 'amber',
    bgClass: 'bg-amber-500/10 border-amber-500/30',
    textClass: 'text-amber-400',
    dotClass: 'bg-amber-400',
  },
  disconnected: {
    label: 'Disconnected',
    color: 'red',
    bgClass: 'bg-red-500/10 border-red-500/30',
    textClass: 'text-red-400',
    dotClass: 'bg-red-400',
  },
};

const categoryColorMap: Record<string, string> = {
  sky: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
  violet: 'bg-violet-500/20 text-violet-400 border-violet-500/30',
  amber: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  emerald: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  cyan: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
};

export function BrainStatus() {
  const brain = useBrainState();
  const health = healthConfig[brain.connectionHealth];
  const maxQueries = Math.max(...brain.topCategories.map((c) => c.queries));

  return (
    <div className="space-y-6">
      {/* Connection Health Banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`p-4 rounded-lg border flex items-center justify-between ${health.bgClass}`}
      >
        <div className="flex items-center gap-3">
          <motion.div
            className={`w-3 h-3 rounded-full ${health.dotClass}`}
            animate={
              brain.connectionHealth === 'connected'
                ? { scale: [1, 1.3, 1], opacity: [1, 0.7, 1] }
                : {}
            }
            transition={{ duration: 2, repeat: Infinity }}
          />
          <Brain className={health.textClass} size={20} />
          <div>
            <span className={`font-medium ${health.textClass}`}>
              Brain Link: {health.label}
            </span>
            <p className="text-xs text-zinc-500 mt-0.5">
              {brain.connectionHealth === 'connected'
                ? `Relay latency: ${brain.relayLatency.toFixed(0)}ms`
                : brain.connectionHealth === 'degraded'
                  ? `High latency: ${brain.relayLatency.toFixed(0)}ms`
                  : 'Enable contribution to connect'}
            </p>
          </div>
        </div>
        {brain.connectionHealth !== 'disconnected' && (
          <div className="flex items-center gap-1 text-xs text-zinc-500">
            <Link2 size={12} />
            <span>relay &rarr; brain</span>
          </div>
        )}
      </motion.div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="bg-gradient-to-br from-sky-500/20 to-sky-600/10 border border-sky-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <Activity className="text-sky-400" size={22} />
                <span className="text-xs text-sky-400/70">Today</span>
              </div>
              <p className="text-3xl font-bold text-white stat-value">
                {brain.operationsToday.toLocaleString()}
              </p>
              <p className="text-sm text-sky-400 mt-1">Brain Operations</p>
            </CardBody>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="bg-gradient-to-br from-emerald-500/20 to-emerald-600/10 border border-emerald-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <Coins className="text-emerald-400" size={22} />
                <span className="text-xs text-emerald-400/70">Earned</span>
              </div>
              <p className="text-3xl font-bold text-white stat-value">
                {brain.ruvEarnedToday.toFixed(2)}
              </p>
              <p className="text-sm text-emerald-400 mt-1">rUv from Brain</p>
            </CardBody>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="bg-gradient-to-br from-violet-500/20 to-violet-600/10 border border-violet-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <Clock className="text-violet-400" size={22} />
                <span className="text-xs text-violet-400/70">Epoch</span>
              </div>
              <p className="text-3xl font-bold text-white stat-value">
                {brain.halvingEpoch}
              </p>
              <p className="text-sm text-violet-400 mt-1">Halving Epoch</p>
            </CardBody>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="bg-gradient-to-br from-cyan-500/20 to-cyan-600/10 border border-cyan-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <BookOpen className="text-cyan-400" size={22} />
                <span className="text-xs text-cyan-400/70">Total</span>
              </div>
              <p className="text-3xl font-bold text-white stat-value">
                {(brain.totalKnowledgeEntries / 1000).toFixed(1)}k
              </p>
              <p className="text-sm text-cyan-400 mt-1">Knowledge Entries</p>
            </CardBody>
          </Card>
        </motion.div>
      </div>

      {/* Halving Epoch Progress */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Zap className="text-violet-400" size={18} />
            Halving Epoch Progress
          </h3>
          <span className="text-sm text-zinc-400">
            Epoch {brain.halvingEpoch} of 20
          </span>
        </div>
        <div className="mb-3">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-zinc-400">Progress to next halving</span>
            <span className="text-violet-400">
              {(brain.epochProgress * 100).toFixed(0)}%
            </span>
          </div>
          <Progress
            value={brain.epochProgress * 100}
            maxValue={100}
            classNames={{
              indicator: 'bg-gradient-to-r from-violet-500 to-pink-500',
              track: 'bg-zinc-800',
            }}
          />
        </div>
        <div className="grid grid-cols-3 gap-4 mt-4">
          <div className="text-center p-3 rounded-lg bg-zinc-800/50">
            <p className="text-lg font-bold text-white">1.0x</p>
            <p className="text-xs text-zinc-400">Current Rate</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-zinc-800/50">
            <p className="text-lg font-bold text-white">0.5x</p>
            <p className="text-xs text-zinc-400">Next Epoch Rate</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-zinc-800/50">
            <p className="text-lg font-bold text-white">10B</p>
            <p className="text-xs text-zinc-400">Max Supply rUv</p>
          </div>
        </div>
      </motion.div>

      {/* Top Knowledge Categories */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="text-sky-400" size={18} />
          Top Knowledge Categories
        </h3>
        <div className="space-y-3">
          {brain.topCategories.map((cat, idx) => (
            <div key={cat.name} className="flex items-center gap-3">
              <span className="text-xs text-zinc-500 w-4">{idx + 1}</span>
              <div className="flex-1">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-zinc-300">{cat.name}</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full border ${categoryColorMap[cat.color]}`}
                  >
                    {cat.queries.toLocaleString()} queries
                  </span>
                </div>
                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${(cat.queries / maxQueries) * 100}%` }}
                    transition={{ duration: 0.5, delay: idx * 0.1 }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

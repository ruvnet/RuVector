import { useState, useEffect, useMemo } from 'react';
import { Card, CardBody, Progress } from '@heroui/react';
import { motion } from 'framer-motion';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  AreaChart,
  Area,
} from 'recharts';
import {
  TrendingUp,
  Users,
  Coins,
  Shield,
  Award,
  Gauge,
  CircleDollarSign,
  Gift,
} from 'lucide-react';
import { useNetworkStore } from '../../stores/networkStore';

// Economics model types
interface ReputationTier {
  name: string;
  minScore: number;
  color: string;
  count: number;
  benefits: string;
}

interface SupplyStats {
  totalMinted: number;
  maxSupply: number;
  contributorPool: number;
  treasury: number;
  protocolFund: number;
  founderPool: number;
}

interface LeaderboardEntry {
  rank: number;
  name: string;
  ruvEarned: number;
  tasks: number;
  tier: string;
}

interface EconomicsState {
  contributionMultiplier: number;
  multiplierCurve: { x: number; y: number }[];
  reputationTiers: ReputationTier[];
  supply: SupplyStats;
  freeReadsToday: number;
  freeReadsLimit: number;
  leaderboard: LeaderboardEntry[];
  velocity: number;
  utilization: number;
  stability: number;
}

function useEconomicsState(): EconomicsState {
  const { stats, credits } = useNetworkStore();
  const [economics, setEconomics] = useState<EconomicsState>(() => ({
    contributionMultiplier: 1.0,
    multiplierCurve: Array.from({ length: 20 }, (_, i) => ({
      x: i * 5,
      y: Math.min(3.0, 1.0 + Math.log1p(i * 5 * 0.02) * 1.2),
    })),
    reputationTiers: [
      { name: 'Observer', minScore: 0, color: '#71717a', count: 2841, benefits: 'Free reads only' },
      { name: 'Contributor', minScore: 0.3, color: '#38bdf8', count: 1592, benefits: '+1.2x multiplier' },
      { name: 'Builder', minScore: 0.5, color: '#a78bfa', count: 823, benefits: '+1.8x, priority queue' },
      { name: 'Architect', minScore: 0.7, color: '#fbbf24', count: 241, benefits: '+2.5x, governance' },
      { name: 'Guardian', minScore: 0.9, color: '#34d399', count: 47, benefits: '+3.0x, all access' },
    ],
    supply: {
      totalMinted: 847_293_100,
      maxSupply: 10_000_000_000,
      contributorPool: 593_105_170,
      treasury: 127_093_965,
      protocolFund: 84_729_310,
      founderPool: 42_364_655,
    },
    freeReadsToday: 14,
    freeReadsLimit: 20,
    leaderboard: [
      { rank: 1, name: 'node-7f3a...c2e1', ruvEarned: 12847.5, tasks: 28341, tier: 'Guardian' },
      { rank: 2, name: 'node-b2d1...9f4a', ruvEarned: 9432.1, tasks: 21203, tier: 'Architect' },
      { rank: 3, name: 'node-e5c8...1b7d', ruvEarned: 7891.3, tasks: 17654, tier: 'Architect' },
      { rank: 4, name: 'node-a1f9...d3e5', ruvEarned: 5234.8, tasks: 12089, tier: 'Builder' },
      { rank: 5, name: 'node-c4b6...8a2f', ruvEarned: 3912.4, tasks: 8921, tier: 'Builder' },
    ],
    velocity: 0.42,
    utilization: 0.68,
    stability: 0.91,
  }));

  useEffect(() => {
    const interval = setInterval(() => {
      setEconomics((prev) => {
        const newMultiplier = Math.min(
          3.0,
          1.0 + Math.log1p(credits.earned * 0.02) * 1.2
        );
        return {
          ...prev,
          contributionMultiplier: Math.round(newMultiplier * 100) / 100,
          supply: {
            ...prev.supply,
            totalMinted: prev.supply.totalMinted + Math.floor(Math.random() * 100),
          },
          freeReadsToday: Math.min(prev.freeReadsLimit, prev.freeReadsToday + (Math.random() > 0.95 ? 1 : 0)),
          velocity: 0.42 + Math.sin(Date.now() / 10000) * 0.05,
          utilization: Math.min(1.0, 0.68 + stats.tasksCompleted * 0.0001),
        };
      });
    }, 3000);
    return () => clearInterval(interval);
  }, [credits.earned, stats.tasksCompleted]);

  return economics;
}

const SUPPLY_COLORS = ['#38bdf8', '#a78bfa', '#06b6d4', '#fbbf24'];

const tierColorMap: Record<string, string> = {
  Observer: 'text-zinc-400',
  Contributor: 'text-sky-400',
  Builder: 'text-violet-400',
  Architect: 'text-amber-400',
  Guardian: 'text-emerald-400',
};

export function EconomicsOverview() {
  const eco = useEconomicsState();

  const supplyPieData = useMemo(
    () => [
      { name: 'Contributors (70%)', value: eco.supply.contributorPool },
      { name: 'Treasury (15%)', value: eco.supply.treasury },
      { name: 'Protocol (10%)', value: eco.supply.protocolFund },
      { name: 'Founders (5%)', value: eco.supply.founderPool },
    ],
    [eco.supply]
  );

  const tierBarData = useMemo(
    () =>
      eco.reputationTiers.map((t) => ({
        name: t.name,
        count: t.count,
        fill: t.color,
      })),
    [eco.reputationTiers]
  );

  const mintPercentage =
    (eco.supply.totalMinted / eco.supply.maxSupply) * 100;

  return (
    <div className="space-y-6">
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="bg-gradient-to-br from-sky-500/20 to-sky-600/10 border border-sky-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <TrendingUp className="text-sky-400" size={22} />
                <span className="text-xs text-sky-400/70">Your Rate</span>
              </div>
              <p className="text-3xl font-bold text-white stat-value">
                {eco.contributionMultiplier.toFixed(2)}x
              </p>
              <p className="text-sm text-sky-400 mt-1">
                Contribution Multiplier
              </p>
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
                <Gauge className="text-emerald-400" size={22} />
                <span className="text-xs text-emerald-400/70">Health</span>
              </div>
              <p className="text-3xl font-bold text-white stat-value">
                {(eco.stability * 100).toFixed(0)}%
              </p>
              <p className="text-sm text-emerald-400 mt-1">
                Economic Stability
              </p>
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
                <CircleDollarSign className="text-violet-400" size={22} />
                <span className="text-xs text-violet-400/70">Velocity</span>
              </div>
              <p className="text-3xl font-bold text-white stat-value">
                {(eco.velocity * 100).toFixed(0)}%
              </p>
              <p className="text-sm text-violet-400 mt-1">rUv Circulation</p>
            </CardBody>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="bg-gradient-to-br from-amber-500/20 to-amber-600/10 border border-amber-500/30">
            <CardBody className="p-5">
              <div className="flex items-center justify-between mb-2">
                <Gift className="text-amber-400" size={22} />
                <span className="text-xs text-amber-400/70">Free Tier</span>
              </div>
              <p className="text-3xl font-bold text-white stat-value">
                {eco.freeReadsToday}/{eco.freeReadsLimit}
              </p>
              <p className="text-sm text-amber-400 mt-1">Free Reads Today</p>
            </CardBody>
          </Card>
        </motion.div>
      </div>

      {/* Middle Row: Supply + Multiplier Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* rUv Supply Distribution */}
        <motion.div
          className="crystal-card p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Coins className="text-sky-400" size={18} />
            rUv Supply Distribution
          </h3>
          <div className="flex items-center gap-6">
            <div className="w-40 h-40 flex-shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={supplyPieData}
                    innerRadius={45}
                    outerRadius={70}
                    paddingAngle={3}
                    dataKey="value"
                    stroke="none"
                  >
                    {supplyPieData.map((_, idx) => (
                      <Cell key={idx} fill={SUPPLY_COLORS[idx]} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex-1 space-y-2">
              {supplyPieData.map((entry, idx) => (
                <div
                  key={entry.name}
                  className="flex items-center justify-between text-sm"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: SUPPLY_COLORS[idx] }}
                    />
                    <span className="text-zinc-300">{entry.name}</span>
                  </div>
                  <span className="text-zinc-400 font-mono text-xs">
                    {(entry.value / 1_000_000).toFixed(1)}M
                  </span>
                </div>
              ))}
              <div className="pt-2 border-t border-white/10">
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-400">Minted / Max</span>
                  <span className="text-white font-medium">
                    {mintPercentage.toFixed(2)}%
                  </span>
                </div>
                <Progress
                  value={mintPercentage}
                  maxValue={100}
                  size="sm"
                  classNames={{
                    indicator:
                      'bg-gradient-to-r from-sky-500 via-violet-500 to-cyan-500',
                    track: 'bg-zinc-800',
                  }}
                  className="mt-2"
                />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Contribution Multiplier Curve */}
        <motion.div
          className="crystal-card p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="text-violet-400" size={18} />
            Contribution Multiplier Curve
          </h3>
          <p className="text-xs text-zinc-500 mb-3">
            Earn more rUv as your contribution score grows. Logarithmic curve
            rewards early participants.
          </p>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={eco.multiplierCurve}>
                <defs>
                  <linearGradient
                    id="multiplierGrad"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="x"
                  tick={{ fill: '#71717a', fontSize: 10 }}
                  axisLine={{ stroke: '#27272a' }}
                  tickLine={false}
                  label={{
                    value: 'Contribution Score',
                    position: 'insideBottom',
                    offset: -2,
                    fill: '#71717a',
                    fontSize: 10,
                  }}
                />
                <YAxis
                  tick={{ fill: '#71717a', fontSize: 10 }}
                  axisLine={{ stroke: '#27272a' }}
                  tickLine={false}
                  domain={[0.5, 3.5]}
                  label={{
                    value: 'Multiplier',
                    angle: -90,
                    position: 'insideLeft',
                    offset: 10,
                    fill: '#71717a',
                    fontSize: 10,
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(value: number) => [
                    `${value.toFixed(2)}x`,
                    'Multiplier',
                  ]}
                />
                <Area
                  type="monotone"
                  dataKey="y"
                  stroke="#a78bfa"
                  strokeWidth={2}
                  fill="url(#multiplierGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-violet-400" />
            <span className="text-xs text-zinc-400">
              Your current multiplier:{' '}
              <span className="text-violet-400 font-semibold">
                {eco.contributionMultiplier.toFixed(2)}x
              </span>
            </span>
          </div>
        </motion.div>
      </div>

      {/* Bottom Row: Reputation Tiers + Leaderboard */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reputation Tier Distribution */}
        <motion.div
          className="crystal-card p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Shield className="text-amber-400" size={18} />
            Reputation Tier Distribution
          </h3>
          <div className="h-44 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tierBarData} barSize={32}>
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#71717a', fontSize: 10 }}
                  axisLine={{ stroke: '#27272a' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#71717a', fontSize: 10 }}
                  axisLine={{ stroke: '#27272a' }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(value: number) => [
                    value.toLocaleString(),
                    'Nodes',
                  ]}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {tierBarData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.fill} fillOpacity={0.7} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2">
            {eco.reputationTiers.map((tier) => (
              <div
                key={tier.name}
                className="flex items-center justify-between text-sm"
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: tier.color }}
                  />
                  <span className="text-zinc-300">{tier.name}</span>
                  <span className="text-xs text-zinc-600">
                    ({'>'}
                    {(tier.minScore * 100).toFixed(0)}%)
                  </span>
                </div>
                <span className="text-xs text-zinc-500">{tier.benefits}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Top Earners Leaderboard */}
        <motion.div
          className="crystal-card p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Award className="text-emerald-400" size={18} />
            Top Earners
          </h3>
          <div className="space-y-3">
            {eco.leaderboard.map((entry) => (
              <div
                key={entry.rank}
                className="flex items-center gap-3 p-3 rounded-lg bg-zinc-800/50"
              >
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                    entry.rank === 1
                      ? 'bg-amber-500/20 text-amber-400'
                      : entry.rank === 2
                        ? 'bg-zinc-400/20 text-zinc-300'
                        : entry.rank === 3
                          ? 'bg-orange-500/20 text-orange-400'
                          : 'bg-zinc-700/50 text-zinc-500'
                  }`}
                >
                  {entry.rank}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-mono text-zinc-300 truncate">
                    {entry.name}
                  </p>
                  <p className="text-xs text-zinc-500">
                    {entry.tasks.toLocaleString()} tasks
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-emerald-400">
                    {entry.ruvEarned.toLocaleString()} rUv
                  </p>
                  <p
                    className={`text-xs ${tierColorMap[entry.tier] || 'text-zinc-400'}`}
                  >
                    {entry.tier}
                  </p>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 rounded-lg bg-sky-500/10 border border-sky-500/20">
            <div className="flex items-center gap-2">
              <Users className="text-sky-400" size={14} />
              <span className="text-xs text-sky-400">
                {eco.reputationTiers
                  .reduce((sum, t) => sum + t.count, 0)
                  .toLocaleString()}{' '}
                total network participants
              </span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

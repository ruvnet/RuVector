import { Card, CardBody } from '@heroui/react';
import { motion } from 'framer-motion';
import {
  Gift,
  ArrowRight,
  BookOpen,
  PenTool,
  Upload,
  Shield,
  Star,
  CheckCircle2,
  Zap,
  TrendingUp,
  Users,
  Cpu,
} from 'lucide-react';

// Step-by-step how to earn
interface EarnStep {
  step: number;
  title: string;
  description: string;
  icon: React.ReactNode;
  color: string;
}

const earnSteps: EarnStep[] = [
  {
    step: 1,
    title: 'Enable Contribution',
    description:
      'Toggle on compute sharing from the consent widget. Your browser contributes idle CPU cycles to the network.',
    icon: <Cpu size={20} />,
    color: 'sky',
  },
  {
    step: 2,
    title: 'Complete Tasks',
    description:
      'Your node automatically picks up and processes tasks from the network. Each completed task earns rUv.',
    icon: <CheckCircle2 size={20} />,
    color: 'emerald',
  },
  {
    step: 3,
    title: 'Build Reputation',
    description:
      'Consistent, accurate contributions increase your reputation score, unlocking higher tiers and multipliers.',
    icon: <TrendingUp size={20} />,
    color: 'violet',
  },
  {
    step: 4,
    title: 'Contribute Knowledge',
    description:
      'Share knowledge patterns to the Brain. Quality contributions earn bonus rUv through the earn-to-write model.',
    icon: <Upload size={20} />,
    color: 'amber',
  },
];

// Reward table data
interface RewardAction {
  action: string;
  reward: string;
  cost: string;
  frequency: string;
  icon: React.ReactNode;
}

const rewardTable: RewardAction[] = [
  {
    action: 'Read from Brain',
    reward: 'Free',
    cost: '0 rUv',
    frequency: '20/day free, then 0.001 rUv',
    icon: <BookOpen size={16} className="text-sky-400" />,
  },
  {
    action: 'Write to Brain',
    reward: '0.01-0.10 rUv',
    cost: '0 rUv (earn-to-write)',
    frequency: 'Per accepted entry',
    icon: <PenTool size={16} className="text-emerald-400" />,
  },
  {
    action: 'Compute Task',
    reward: '0.001-0.05 rUv',
    cost: 'CPU/GPU time',
    frequency: 'Per completed task',
    icon: <Cpu size={16} className="text-violet-400" />,
  },
  {
    action: 'Uptime Bonus',
    reward: '0.005 rUv/hr',
    cost: 'Stay online',
    frequency: 'Continuous',
    icon: <Zap size={16} className="text-amber-400" />,
  },
  {
    action: 'Quality Bonus',
    reward: 'Up to 3x multiplier',
    cost: 'High accuracy',
    frequency: 'Applied to all earnings',
    icon: <Star size={16} className="text-pink-400" />,
  },
  {
    action: 'Network Referral',
    reward: '5% of referee earnings',
    cost: 'Share your node link',
    frequency: 'Ongoing for 90 days',
    icon: <Users size={16} className="text-cyan-400" />,
  },
];

// Tier benefits comparison
interface TierBenefit {
  tier: string;
  color: string;
  bgClass: string;
  multiplier: string;
  freeReads: string;
  writePriority: string;
  governance: boolean;
  specialAccess: string;
}

const tierBenefits: TierBenefit[] = [
  {
    tier: 'Observer',
    color: 'text-zinc-400',
    bgClass: 'bg-zinc-500/10 border-zinc-500/30',
    multiplier: '1.0x',
    freeReads: '20/day',
    writePriority: 'Standard',
    governance: false,
    specialAccess: 'None',
  },
  {
    tier: 'Contributor',
    color: 'text-sky-400',
    bgClass: 'bg-sky-500/10 border-sky-500/30',
    multiplier: '1.2x',
    freeReads: '50/day',
    writePriority: 'Standard',
    governance: false,
    specialAccess: 'Metrics API',
  },
  {
    tier: 'Builder',
    color: 'text-violet-400',
    bgClass: 'bg-violet-500/10 border-violet-500/30',
    multiplier: '1.8x',
    freeReads: '100/day',
    writePriority: 'Priority',
    governance: false,
    specialAccess: 'Advanced search',
  },
  {
    tier: 'Architect',
    color: 'text-amber-400',
    bgClass: 'bg-amber-500/10 border-amber-500/30',
    multiplier: '2.5x',
    freeReads: 'Unlimited',
    writePriority: 'Priority+',
    governance: true,
    specialAccess: 'Custom models',
  },
  {
    tier: 'Guardian',
    color: 'text-emerald-400',
    bgClass: 'bg-emerald-500/10 border-emerald-500/30',
    multiplier: '3.0x',
    freeReads: 'Unlimited',
    writePriority: 'Instant',
    governance: true,
    specialAccess: 'Full API + governance',
  },
];

export function RewardsGuide() {
  return (
    <div className="space-y-6">
      {/* How to Earn Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="crystal-card p-6">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-emerald-500/20">
              <Gift className="text-emerald-400" size={24} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">How to Earn rUv</h2>
              <p className="text-sm text-zinc-400">
                rUv is the native token of the Edge-Net economy. Everyone starts
                earning for free.
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Step-by-Step Guide */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {earnSteps.map((step, idx) => {
          const colorMap: Record<string, string> = {
            sky: 'from-sky-500/20 to-sky-600/10 border-sky-500/30',
            emerald:
              'from-emerald-500/20 to-emerald-600/10 border-emerald-500/30',
            violet:
              'from-violet-500/20 to-violet-600/10 border-violet-500/30',
            amber: 'from-amber-500/20 to-amber-600/10 border-amber-500/30',
          };
          const iconColorMap: Record<string, string> = {
            sky: 'text-sky-400 bg-sky-500/20',
            emerald: 'text-emerald-400 bg-emerald-500/20',
            violet: 'text-violet-400 bg-violet-500/20',
            amber: 'text-amber-400 bg-amber-500/20',
          };

          return (
            <motion.div
              key={step.step}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <Card
                className={`bg-gradient-to-br ${colorMap[step.color]} border h-full`}
              >
                <CardBody className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${iconColorMap[step.color]}`}
                    >
                      {step.step}
                    </div>
                    {idx < earnSteps.length - 1 && (
                      <ArrowRight
                        size={12}
                        className="text-zinc-600 hidden sm:block"
                      />
                    )}
                  </div>
                  <div className={`mb-3 ${iconColorMap[step.color]} w-fit p-2 rounded-lg`}>
                    {step.icon}
                  </div>
                  <h3 className="font-semibold text-white mb-1">
                    {step.title}
                  </h3>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    {step.description}
                  </p>
                </CardBody>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Reward Actions Table */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Zap className="text-amber-400" size={18} />
          Reward Table
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-zinc-500 font-medium">
                  Action
                </th>
                <th className="text-left py-2 px-3 text-zinc-500 font-medium">
                  Reward
                </th>
                <th className="text-left py-2 px-3 text-zinc-500 font-medium">
                  Cost
                </th>
                <th className="text-left py-2 px-3 text-zinc-500 font-medium">
                  Details
                </th>
              </tr>
            </thead>
            <tbody>
              {rewardTable.map((row) => (
                <tr
                  key={row.action}
                  className="border-b border-white/5 hover:bg-white/5 transition-colors"
                >
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-2">
                      {row.icon}
                      <span className="text-zinc-300">{row.action}</span>
                    </div>
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-emerald-400 font-medium">
                      {row.reward}
                    </span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-zinc-400">{row.cost}</span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-zinc-500 text-xs">
                      {row.frequency}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Tier Benefits Comparison */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield className="text-violet-400" size={18} />
          Tier Benefits Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-zinc-500 font-medium">
                  Tier
                </th>
                <th className="text-center py-2 px-3 text-zinc-500 font-medium">
                  Multiplier
                </th>
                <th className="text-center py-2 px-3 text-zinc-500 font-medium">
                  Free Reads
                </th>
                <th className="text-center py-2 px-3 text-zinc-500 font-medium">
                  Write Priority
                </th>
                <th className="text-center py-2 px-3 text-zinc-500 font-medium">
                  Governance
                </th>
                <th className="text-center py-2 px-3 text-zinc-500 font-medium">
                  Special Access
                </th>
              </tr>
            </thead>
            <tbody>
              {tierBenefits.map((tier) => (
                <tr
                  key={tier.tier}
                  className="border-b border-white/5 hover:bg-white/5 transition-colors"
                >
                  <td className="py-3 px-3">
                    <span className={`font-semibold ${tier.color}`}>
                      {tier.tier}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-center">
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs border ${tier.bgClass} ${tier.color}`}
                    >
                      {tier.multiplier}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-center text-zinc-300">
                    {tier.freeReads}
                  </td>
                  <td className="py-3 px-3 text-center text-zinc-300">
                    {tier.writePriority}
                  </td>
                  <td className="py-3 px-3 text-center">
                    {tier.governance ? (
                      <CheckCircle2 size={16} className="text-emerald-400 mx-auto" />
                    ) : (
                      <span className="text-zinc-600">-</span>
                    )}
                  </td>
                  <td className="py-3 px-3 text-center text-zinc-400 text-xs">
                    {tier.specialAccess}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Key Principles */}
      <motion.div
        className="crystal-card p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Star className="text-pink-400" size={18} />
          Key Economic Principles
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-4 rounded-lg bg-sky-500/10 border border-sky-500/20">
            <h4 className="font-medium text-sky-400 mb-1">
              Always Free to Start
            </h4>
            <p className="text-xs text-zinc-400">
              No upfront cost. Free reads every day. Start earning immediately by
              contributing compute or knowledge.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-violet-500/10 border border-violet-500/20">
            <h4 className="font-medium text-violet-400 mb-1">
              Earn-to-Write Model
            </h4>
            <p className="text-xs text-zinc-400">
              Writing to the Brain is not a cost -- it is a reward. Quality
              knowledge contributions earn rUv.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <h4 className="font-medium text-emerald-400 mb-1">
              Halving Epochs
            </h4>
            <p className="text-xs text-zinc-400">
              Like Bitcoin, mining rewards halve periodically. Early contributors
              earn more rUv per unit of work.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-amber-500/10 border border-amber-500/20">
            <h4 className="font-medium text-amber-400 mb-1">
              Reputation Matters
            </h4>
            <p className="text-xs text-zinc-400">
              Reputation decays over time to prevent gaming. Consistent quality
              contributions keep your multiplier high.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

"use client";

import React from "react";
import AppShell from "../components/AppShell";
import { PageHeader, StatCard, StatusBadge, SiteBadge } from "../components/shared";
import {
  PROMOTIONS,
  AUDIT_EVENTS,
  CAMPAIGNS,
  formatDateTime,
  formatCents,
} from "../data/store";
import {
  Tag,
  CheckCircle2,
  TrendingUp,
  Clock,
  Activity,
  AlertTriangle,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import Link from "next/link";

function RecentActivityItem({ event }: { event: typeof AUDIT_EVENTS[0] }) {
  const actionIcons: Record<string, React.ReactNode> = {
    "promotion.created": <Tag size={14} className="text-blue-500" />,
    "promotion.updated": <Activity size={14} className="text-amber-500" />,
    "promotion.approved": <CheckCircle2 size={14} className="text-emerald-500" />,
    "promotion.submitted": <Clock size={14} className="text-purple-500" />,
    "promotion.suspended": <AlertTriangle size={14} className="text-red-500" />,
    "promotion.activated": <TrendingUp size={14} className="text-emerald-500" />,
    "codes.exported": <Activity size={14} className="text-gray-500" />,
    "user.role_changed": <Activity size={14} className="text-indigo-500" />,
  };

  return (
    <div className="flex items-start gap-3 py-3 border-b border-gray-100 last:border-0">
      <div className="mt-0.5 p-1.5 rounded-lg bg-gray-50">
        {actionIcons[event.action] || <Activity size={14} className="text-gray-400" />}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-900">
          <span className="font-semibold">{event.actor}</span>{" "}
          <span className="text-gray-500">{event.action.replace(".", " ").replace("_", " ")}</span>{" "}
          <span className="font-medium">{event.entityName}</span>
        </p>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-xs text-gray-400">{formatDateTime(event.timestamp)}</span>
          <SiteBadge site={event.site} />
          {event.aiAssisted && (
            <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-purple-600 bg-purple-50 px-1.5 py-0.5 rounded">
              <Sparkles size={10} /> AI
            </span>
          )}
          {event.automatic && (
            <span className="text-[10px] font-semibold text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded">
              Auto
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const activePromos = PROMOTIONS.filter((p) => p.status === "active").length;
  const pendingApprovals = PROMOTIONS.filter((p) => p.status === "in_review").length;
  const activeCampaigns = CAMPAIGNS.filter((c) => c.status === "active").length;
  const totalRedemptions = PROMOTIONS.reduce(
    (sum, p) => sum + (p.limits?.usedCount || 0),
    0
  );

  const upcomingPromos = PROMOTIONS.filter(
    (p) => p.status === "approved" || (p.status === "active" && new Date(p.endsAt) > new Date())
  ).slice(0, 5);

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="Dashboard"
          subtitle="Promotion Engine overview across all four sites"
        />

        {/* Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <StatCard
            label="Active Promotions"
            value={activePromos}
            change="+2 this week"
            changeType="up"
            icon={<Tag size={20} className="text-bass-green" />}
          />
          <StatCard
            label="Pending Approvals"
            value={pendingApprovals}
            change="1 urgent"
            changeType="neutral"
            icon={<CheckCircle2 size={20} className="text-amber-500" />}
          />
          <StatCard
            label="Active Campaigns"
            value={activeCampaigns}
            change="3 planned"
            changeType="neutral"
            icon={<TrendingUp size={20} className="text-blue-500" />}
          />
          <StatCard
            label="Total Redemptions"
            value={totalRedemptions.toLocaleString()}
            change="+12.4% vs last month"
            changeType="up"
            icon={<Activity size={20} className="text-purple-500" />}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Activity */}
          <div className="lg:col-span-2 card">
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
              <h2 className="text-base font-semibold text-gray-900">Recent Activity</h2>
              <Link href="/audit" className="text-sm text-bass-gold-dark font-medium hover:underline flex items-center gap-1">
                View all <ArrowRight size={14} />
              </Link>
            </div>
            <div className="px-5 py-2">
              {AUDIT_EVENTS.slice(0, 8).map((event) => (
                <RecentActivityItem key={event.id} event={event} />
              ))}
            </div>
          </div>

          {/* Upcoming / Active Promotions */}
          <div className="card">
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
              <h2 className="text-base font-semibold text-gray-900">Active & Upcoming</h2>
              <Link href="/promotions" className="text-sm text-bass-gold-dark font-medium hover:underline flex items-center gap-1">
                View all <ArrowRight size={14} />
              </Link>
            </div>
            <div className="px-5 py-3 space-y-3">
              {upcomingPromos.map((promo) => (
                <Link
                  key={promo.id}
                  href={`/promotions/${promo.id}`}
                  className="block p-3 rounded-lg border border-gray-100 hover:border-bass-gold/30 hover:bg-bass-cream/30 transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <span className="text-sm font-semibold text-gray-900">{promo.name}</span>
                    <StatusBadge status={promo.status} />
                  </div>
                  <div className="flex items-center gap-1.5 mb-2">
                    {promo.sites.map((s) => (
                      <SiteBadge key={s} site={s} />
                    ))}
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{promo.templateCode} · v{promo.version}</span>
                    {promo.limits?.usedCount !== undefined && (
                      <span>{promo.limits.usedCount.toLocaleString()} used</span>
                    )}
                  </div>
                  {promo.limits?.totalRedemptions && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-100 rounded-full h-1.5">
                        <div
                          className="bg-bass-gold rounded-full h-1.5 transition-all"
                          style={{
                            width: `${Math.min(100, ((promo.limits.usedCount || 0) / promo.limits.totalRedemptions) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mt-6 card p-5">
          <h2 className="text-base font-semibold text-gray-900 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Link href="/promotions/new" className="flex flex-col items-center gap-2 p-4 rounded-lg border border-gray-200 hover:border-bass-gold/40 hover:bg-bass-cream/30 transition-colors">
              <Tag size={24} className="text-bass-green" />
              <span className="text-sm font-medium text-gray-700">New Promotion</span>
            </Link>
            <Link href="/simulator" className="flex flex-col items-center gap-2 p-4 rounded-lg border border-gray-200 hover:border-bass-gold/40 hover:bg-bass-cream/30 transition-colors">
              <Activity size={24} className="text-blue-500" />
              <span className="text-sm font-medium text-gray-700">Run Simulator</span>
            </Link>
            <Link href="/approvals" className="flex flex-col items-center gap-2 p-4 rounded-lg border border-gray-200 hover:border-bass-gold/40 hover:bg-bass-cream/30 transition-colors">
              <CheckCircle2 size={24} className="text-amber-500" />
              <span className="text-sm font-medium text-gray-700">Review Approvals</span>
            </Link>
            <Link href="/ai-assistant" className="flex flex-col items-center gap-2 p-4 rounded-lg border border-gray-200 hover:border-bass-gold/40 hover:bg-bass-cream/30 transition-colors">
              <Sparkles size={24} className="text-purple-500" />
              <span className="text-sm font-medium text-gray-700">AI Assistant</span>
            </Link>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

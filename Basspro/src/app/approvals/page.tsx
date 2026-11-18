"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, StatusBadge, SiteBadge } from "../../components/shared";
import { PROMOTIONS, formatDate, formatCents } from "../../data/store";
import {
  CheckCircle2,
  XCircle,
  Eye,
  ArrowRight,
  Minus,
  Plus,
  AlertTriangle,
  FileText,
  FlaskConical,
} from "lucide-react";
import Link from "next/link";

export default function ApprovalsPage() {
  const [selectedPromo, setSelectedPromo] = useState<string | null>("P-BF25");
  const pendingPromos = PROMOTIONS.filter((p) => p.status === "in_review");
  const recentlyApproved = PROMOTIONS.filter((p) => p.status === "approved");
  const selected = PROMOTIONS.find((p) => p.id === selectedPromo);

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="Approvals"
          subtitle={`${pendingPromos.length} promotions pending review`}
        />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Queue */}
          <div className="card">
            <div className="px-4 py-3 border-b border-gray-100">
              <h3 className="text-sm font-semibold text-gray-900">Approval Queue</h3>
            </div>
            <div className="divide-y divide-gray-50">
              {pendingPromos.map((promo) => (
                <button
                  key={promo.id}
                  onClick={() => setSelectedPromo(promo.id)}
                  className={`w-full text-left p-4 hover:bg-gray-50 transition-colors ${
                    selectedPromo === promo.id ? "bg-bass-cream/50 border-l-2 border-bass-gold" : ""
                  }`}
                >
                  <div className="flex items-start justify-between mb-1">
                    <span className="text-sm font-semibold text-gray-900">{promo.name}</span>
                    <StatusBadge status={promo.status} />
                  </div>
                  <div className="flex gap-1 mb-1.5">
                    {promo.sites.map((s) => <SiteBadge key={s} site={s} />)}
                  </div>
                  <p className="text-xs text-gray-500">
                    v{promo.version} · by {promo.createdBy} · {formatDate(promo.updatedAt)}
                  </p>
                </button>
              ))}
              {pendingPromos.length === 0 && (
                <div className="p-8 text-center text-sm text-gray-400">
                  No promotions pending review
                </div>
              )}

              {recentlyApproved.length > 0 && (
                <>
                  <div className="px-4 py-2 bg-gray-50">
                    <p className="text-xs font-semibold text-gray-500">Recently Approved</p>
                  </div>
                  {recentlyApproved.map((promo) => (
                    <button
                      key={promo.id}
                      onClick={() => setSelectedPromo(promo.id)}
                      className={`w-full text-left p-4 hover:bg-gray-50 transition-colors ${
                        selectedPromo === promo.id ? "bg-bass-cream/50 border-l-2 border-bass-gold" : ""
                      }`}
                    >
                      <div className="flex items-start justify-between mb-1">
                        <span className="text-sm font-semibold text-gray-900">{promo.name}</span>
                        <StatusBadge status={promo.status} />
                      </div>
                      <div className="flex gap-1">
                        {promo.sites.map((s) => <SiteBadge key={s} site={s} />)}
                      </div>
                    </button>
                  ))}
                </>
              )}
            </div>
          </div>

          {/* Detail */}
          <div className="lg:col-span-2 space-y-4">
            {selected ? (
              <>
                {/* Header */}
                <div className="card p-5">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h2 className="text-lg font-bold text-gray-900">{selected.name}</h2>
                      <p className="text-sm text-gray-500">{selected.id} · Version {selected.version} · {selected.templateCode}</p>
                    </div>
                    {selected.status === "in_review" && (
                      <div className="flex gap-2">
                        <button className="btn-primary">
                          <CheckCircle2 size={14} /> Approve
                        </button>
                        <button className="btn-danger">
                          <XCircle size={14} /> Reject
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="flex gap-1.5 mb-3">
                    {selected.sites.map((s) => <SiteBadge key={s} site={s} />)}
                  </div>
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-xs text-gray-500">Reward</p>
                      <p className="font-semibold">
                        {selected.reward.kind === "PERCENT" ? `${selected.reward.value}% off` : formatCents(selected.reward.value) + " off"}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Priority</p>
                      <p className="font-semibold font-mono">{selected.priority}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Combination</p>
                      <p className="font-semibold">{selected.combinationMode}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Author</p>
                      <p className="font-semibold">{selected.createdBy}</p>
                    </div>
                  </div>
                </div>

                {/* Version Diff */}
                {selected.version > 1 && (
                  <div className="card p-5">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <FileText size={16} className="text-blue-500" /> Version Diff (v{selected.version - 1} → v{selected.version})
                    </h3>
                    <div className="space-y-2 font-mono text-xs">
                      <div className="flex items-center gap-2 p-2 rounded bg-red-50 text-red-700">
                        <Minus size={12} /> <span>percent: 10%</span>
                      </div>
                      <div className="flex items-center gap-2 p-2 rounded bg-emerald-50 text-emerald-700">
                        <Plus size={12} /> <span>percent: {selected.reward.value}%</span>
                      </div>
                      <div className="flex items-center gap-2 p-2 rounded bg-red-50 text-red-700">
                        <Minus size={12} /> <span>maxDiscount: $40.00</span>
                      </div>
                      <div className="flex items-center gap-2 p-2 rounded bg-emerald-50 text-emerald-700">
                        <Plus size={12} /> <span>maxDiscount: {selected.reward.maxDiscount ? formatCents(selected.reward.maxDiscount) : "none"}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Simulator Evidence */}
                <div className="card p-5">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <FlaskConical size={16} className="text-purple-500" /> Simulator Evidence
                  </h3>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="p-4 rounded-lg bg-gray-50 border border-gray-100 text-center">
                      <p className="text-xs text-gray-500 mb-1">Live Today</p>
                      <p className="text-xl font-bold text-gray-900">$318.44</p>
                    </div>
                    <div className="p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-center">
                      <p className="text-xs text-emerald-600 mb-1">With This Draft</p>
                      <p className="text-xl font-bold text-emerald-700">$306.74</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-2.5 rounded-lg bg-emerald-50 border border-emerald-100">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 size={14} className="text-emerald-500" />
                        <span className="text-sm font-medium text-emerald-800">{selected.name}</span>
                      </div>
                      <span className="text-sm font-bold text-emerald-700">−$26.00</span>
                    </div>
                    <div className="flex items-center justify-between p-2.5 rounded-lg bg-emerald-50 border border-emerald-100">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 size={14} className="text-emerald-500" />
                        <span className="text-sm font-medium text-emerald-800">CLUB 5% stacked</span>
                      </div>
                      <span className="text-sm font-bold text-emerald-700">−$11.70</span>
                    </div>
                    <div className="flex items-center justify-between p-2.5 rounded-lg bg-gray-50 border border-gray-100">
                      <div className="flex items-center gap-2">
                        <XCircle size={14} className="text-gray-400" />
                        <span className="text-sm font-medium text-gray-500">$10 off rods</span>
                      </div>
                      <span className="text-xs font-medium text-gray-400">ITEM_ALREADY_PROMOTED</span>
                    </div>
                  </div>
                </div>

                {/* Scenario Results */}
                <div className="card p-5">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500" /> Scenario Results
                  </h3>
                  <div className="flex items-center gap-3 p-3 rounded-lg bg-emerald-50 border border-emerald-200">
                    <CheckCircle2 size={20} className="text-emerald-500" />
                    <div>
                      <p className="text-sm font-bold text-emerald-800">84 of 84 scenarios passed</p>
                      <p className="text-xs text-emerald-600">All integrity checks pass · Last run: Nov 16, 2026</p>
                    </div>
                  </div>
                </div>

                {/* Rejection reason (if rejecting) */}
                {selected.status === "in_review" && (
                  <div className="card p-5">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3">Approval Notes</h3>
                    <textarea
                      placeholder="Add a reason for approval or rejection (required for rejection)..."
                      className="input-field h-24 resize-none"
                    />
                    <div className="flex items-center gap-2 mt-3 p-3 rounded-lg bg-amber-50 border border-amber-200">
                      <AlertTriangle size={14} className="text-amber-500 flex-shrink-0" />
                      <p className="text-xs text-amber-700">
                        You cannot approve a version you authored (DR-035). The author of this version is <strong>{selected.createdBy}</strong>.
                      </p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="card p-16 text-center">
                <Eye size={32} className="text-gray-300 mx-auto mb-3" />
                <p className="text-sm text-gray-400">Select a promotion from the queue to review</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

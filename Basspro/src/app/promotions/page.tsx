"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, StatusBadge, SiteBadge } from "../../components/shared";
import {
  PROMOTIONS,
  formatDate,
  formatCents,
  type PromotionStatus,
} from "../../data/store";
import {
  Plus,
  Search,
  Filter,
  ArrowUpDown,
  MoreHorizontal,
  Eye,
  Copy,
  Pause,
  Archive,
  ChevronDown,
} from "lucide-react";
import Link from "next/link";

const STATUS_FILTERS: { label: string; value: PromotionStatus | "all" }[] = [
  { label: "All", value: "all" },
  { label: "Active", value: "active" },
  { label: "In Review", value: "in_review" },
  { label: "Approved", value: "approved" },
  { label: "Draft", value: "draft" },
  { label: "Suspended", value: "suspended" },
  { label: "Ended", value: "ended" },
];

export default function PromotionsPage() {
  const [statusFilter, setStatusFilter] = useState<PromotionStatus | "all">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortField, setSortField] = useState<"name" | "priority" | "updatedAt">("updatedAt");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const filtered = PROMOTIONS.filter((p) => {
    if (statusFilter !== "all" && p.status !== statusFilter) return false;
    if (searchQuery && !p.name.toLowerCase().includes(searchQuery.toLowerCase()) && !p.id.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  }).sort((a, b) => {
    const dir = sortDir === "asc" ? 1 : -1;
    if (sortField === "name") return a.name.localeCompare(b.name) * dir;
    if (sortField === "priority") return (a.priority - b.priority) * dir;
    return (new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime()) * dir;
  });

  const toggleSort = (field: typeof sortField) => {
    if (sortField === field) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="Promotions"
          subtitle={`${PROMOTIONS.length} promotions across all sites`}
          actions={
            <Link href="/promotions/new" className="btn-primary">
              <Plus size={16} /> New Promotion
            </Link>
          }
        />

        {/* Filters */}
        <div className="card mb-6">
          <div className="p-4 flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search by name or ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input-field pl-9"
              />
            </div>
            <div className="flex gap-2 flex-wrap">
              {STATUS_FILTERS.map((f) => (
                <button
                  key={f.value}
                  onClick={() => setStatusFilter(f.value)}
                  className={`px-3 py-2 text-xs font-semibold rounded-lg border transition-colors ${
                    statusFilter === f.value
                      ? "bg-bass-green text-white border-bass-green"
                      : "bg-white text-gray-600 border-gray-200 hover:bg-gray-50"
                  }`}
                >
                  {f.label}
                  {f.value !== "all" && (
                    <span className="ml-1.5 opacity-70">
                      {PROMOTIONS.filter((p) => p.status === f.value).length}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="table-header">
                    <button onClick={() => toggleSort("name")} className="flex items-center gap-1 hover:text-gray-700">
                      Promotion <ArrowUpDown size={12} />
                    </button>
                  </th>
                  <th className="table-header">Status</th>
                  <th className="table-header">Template</th>
                  <th className="table-header">Sites</th>
                  <th className="table-header">
                    <button onClick={() => toggleSort("priority")} className="flex items-center gap-1 hover:text-gray-700">
                      Priority <ArrowUpDown size={12} />
                    </button>
                  </th>
                  <th className="table-header">Reward</th>
                  <th className="table-header">Usage</th>
                  <th className="table-header">
                    <button onClick={() => toggleSort("updatedAt")} className="flex items-center gap-1 hover:text-gray-700">
                      Updated <ArrowUpDown size={12} />
                    </button>
                  </th>
                  <th className="table-header w-10"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filtered.map((promo) => (
                  <tr key={promo.id} className="hover:bg-gray-50/50 transition-colors">
                    <td className="table-cell">
                      <Link href={`/promotions/${promo.id}`} className="group">
                        <div className="font-semibold text-gray-900 group-hover:text-bass-gold-dark transition-colors">
                          {promo.name}
                        </div>
                        <div className="text-xs text-gray-400 mt-0.5">
                          {promo.id} · v{promo.version}
                          {promo.campaignName && (
                            <span className="ml-1.5 text-bass-olive">· {promo.campaignName}</span>
                          )}
                        </div>
                      </Link>
                    </td>
                    <td className="table-cell">
                      <StatusBadge status={promo.status} />
                    </td>
                    <td className="table-cell">
                      <span className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded">
                        {promo.templateCode}
                      </span>
                    </td>
                    <td className="table-cell">
                      <div className="flex gap-1 flex-wrap">
                        {promo.sites.map((s) => (
                          <SiteBadge key={s} site={s} />
                        ))}
                      </div>
                    </td>
                    <td className="table-cell font-mono text-xs">{promo.priority}</td>
                    <td className="table-cell">
                      <span className="text-xs">
                        {promo.reward.kind === "PERCENT"
                          ? `${promo.reward.value}% off`
                          : promo.reward.kind === "AMOUNT_OFF"
                          ? `${formatCents(promo.reward.value)} off`
                          : promo.reward.kind === "FIXED_PRICE"
                          ? `${formatCents(promo.reward.value)} fixed`
                          : "Free item"}
                        {promo.reward.maxDiscount && (
                          <span className="text-gray-400 ml-1">
                            (max {formatCents(promo.reward.maxDiscount)})
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="table-cell">
                      {promo.limits?.totalRedemptions ? (
                        <div>
                          <div className="text-xs font-medium">
                            {(promo.limits.usedCount || 0).toLocaleString()} / {promo.limits.totalRedemptions.toLocaleString()}
                          </div>
                          <div className="w-16 bg-gray-100 rounded-full h-1 mt-1">
                            <div
                              className="bg-bass-gold rounded-full h-1"
                              style={{
                                width: `${Math.min(100, ((promo.limits.usedCount || 0) / promo.limits.totalRedemptions) * 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      ) : (
                        <span className="text-xs text-gray-400">No limit</span>
                      )}
                    </td>
                    <td className="table-cell text-xs text-gray-500">
                      {formatDate(promo.updatedAt)}
                    </td>
                    <td className="table-cell">
                      <div className="relative group">
                        <button className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-400">
                          <MoreHorizontal size={16} />
                        </button>
                        <div className="absolute right-0 top-full mt-1 w-40 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-10 hidden group-hover:block">
                          <Link href={`/promotions/${promo.id}`} className="flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50">
                            <Eye size={14} /> View Details
                          </Link>
                          <button className="flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 w-full text-left">
                            <Copy size={14} /> Clone
                          </button>
                          {promo.status === "active" && (
                            <button className="flex items-center gap-2 px-3 py-2 text-sm text-amber-600 hover:bg-amber-50 w-full text-left">
                              <Pause size={14} /> Pause
                            </button>
                          )}
                          <button className="flex items-center gap-2 px-3 py-2 text-sm text-gray-500 hover:bg-gray-50 w-full text-left">
                            <Archive size={14} /> Archive
                          </button>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-3 border-t border-gray-100 flex items-center justify-between text-sm text-gray-500">
            <span>Showing {filtered.length} of {PROMOTIONS.length} promotions</span>
            <div className="flex gap-1">
              <button className="px-3 py-1 rounded border border-gray-200 hover:bg-gray-50 text-xs font-medium">Previous</button>
              <button className="px-3 py-1 rounded bg-bass-green text-white text-xs font-medium">1</button>
              <button className="px-3 py-1 rounded border border-gray-200 hover:bg-gray-50 text-xs font-medium">Next</button>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

"use client";

import React from "react";
import { useParams } from "next/navigation";
import AppShell from "../../../components/AppShell";
import { PageHeader, StatusBadge, SiteBadge, MoneyText } from "../../../components/shared";
import {
  PROMOTIONS,
  AUDIT_EVENTS,
  formatDate,
  formatDateTime,
  formatCents,
} from "../../../data/store";
import {
  ArrowLeft,
  Edit,
  Copy,
  Pause,
  Send,
  Clock,
  User,
  Tag,
  Shield,
  Layers,
  Target,
  Calendar,
  Hash,
  AlertTriangle,
  CheckCircle2,
  History,
} from "lucide-react";
import Link from "next/link";

export default function PromotionDetailPage() {
  const params = useParams();
  const promo = PROMOTIONS.find((p) => p.id === params.id);

  if (!promo) {
    return (
      <AppShell>
        <div className="p-6">
          <p className="text-gray-500">Promotion not found.</p>
          <Link href="/promotions" className="text-bass-gold-dark hover:underline mt-2 inline-block">
            ← Back to Promotions
          </Link>
        </div>
      </AppShell>
    );
  }

  const relatedEvents = AUDIT_EVENTS.filter((e) => e.entityId === promo.id);

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1200px] mx-auto">
        {/* Breadcrumb */}
        <Link href="/promotions" className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft size={14} /> Back to Promotions
        </Link>

        <PageHeader
          title={promo.name}
          subtitle={`${promo.id} · Version ${promo.version} · Template ${promo.templateCode}`}
          actions={
            <div className="flex gap-2">
              {promo.status === "draft" && (
                <button className="btn-primary"><Send size={14} /> Submit for Review</button>
              )}
              {promo.status === "active" && (
                <button className="btn-secondary"><Pause size={14} /> Pause</button>
              )}
              <button className="btn-secondary"><Edit size={14} /> Edit</button>
              <button className="btn-secondary"><Copy size={14} /> Clone</button>
            </div>
          }
        />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Status & Overview */}
            <div className="card p-5">
              <div className="flex items-center gap-3 mb-4">
                <StatusBadge status={promo.status} />
                <div className="flex gap-1.5">
                  {promo.sites.map((s) => (
                    <SiteBadge key={s} site={s} />
                  ))}
                </div>
                {promo.campaignName && (
                  <span className="text-xs text-bass-olive font-medium bg-bass-olive/10 px-2 py-0.5 rounded">
                    {promo.campaignName}
                  </span>
                )}
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-gray-500 mb-1">Template</p>
                  <p className="text-sm font-semibold">{promo.templateCode}</p>
                  <p className="text-xs text-gray-400">{promo.templateName}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-1">Priority</p>
                  <p className="text-sm font-semibold font-mono">{promo.priority}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-1">Combination</p>
                  <p className="text-sm font-semibold">{promo.combinationMode.replace("_", " ")}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-1">Version</p>
                  <p className="text-sm font-semibold">v{promo.version}</p>
                </div>
              </div>
            </div>

            {/* Reward */}
            <div className="card p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Target size={16} className="text-bass-gold" /> Reward (Then)
              </h3>
              <div className="bg-bass-cream/50 rounded-lg p-4 border border-bass-gold/20">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-bass-gold/10">
                    <Tag size={24} className="text-bass-gold-dark" />
                  </div>
                  <div>
                    <p className="text-lg font-bold text-gray-900">
                      {promo.reward.kind === "PERCENT"
                        ? `${promo.reward.value}% Off`
                        : promo.reward.kind === "AMOUNT_OFF"
                        ? `${formatCents(promo.reward.value)} Off`
                        : promo.reward.kind === "FIXED_PRICE"
                        ? `${formatCents(promo.reward.value)} Fixed Price`
                        : "Free Item"}
                    </p>
                    {promo.reward.maxDiscount && (
                      <p className="text-sm text-gray-500">
                        Max discount: {formatCents(promo.reward.maxDiscount)}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Conditions */}
            <div className="card p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Layers size={16} className="text-blue-500" /> Conditions (If)
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {promo.conditions.categories && (
                  <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                    <p className="text-xs text-gray-500 mb-1">Categories</p>
                    <div className="flex flex-wrap gap-1">
                      {promo.conditions.categories.map((c) => (
                        <span key={c} className="text-xs bg-white border border-gray-200 px-2 py-0.5 rounded">{c}</span>
                      ))}
                    </div>
                  </div>
                )}
                {promo.conditions.minQty && (
                  <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                    <p className="text-xs text-gray-500 mb-1">Minimum Quantity</p>
                    <p className="text-sm font-semibold">{promo.conditions.minQty}+</p>
                  </div>
                )}
                {promo.conditions.minSpend && (
                  <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                    <p className="text-xs text-gray-500 mb-1">Minimum Spend</p>
                    <p className="text-sm font-semibold">{formatCents(promo.conditions.minSpend)}</p>
                  </div>
                )}
                {promo.conditions.segments && (
                  <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                    <p className="text-xs text-gray-500 mb-1">Customer Segments</p>
                    <div className="flex flex-wrap gap-1">
                      {promo.conditions.segments.map((s) => (
                        <span key={s} className="text-xs bg-white border border-gray-200 px-2 py-0.5 rounded">{s}</span>
                      ))}
                    </div>
                  </div>
                )}
                {promo.conditions.channels && (
                  <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                    <p className="text-xs text-gray-500 mb-1">Channels</p>
                    <div className="flex flex-wrap gap-1">
                      {promo.conditions.channels.map((ch) => (
                        <span key={ch} className="text-xs bg-white border border-gray-200 px-2 py-0.5 rounded capitalize">{ch}</span>
                      ))}
                    </div>
                  </div>
                )}
                {promo.codes && promo.codes.length > 0 && (
                  <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                    <p className="text-xs text-gray-500 mb-1">Promo Codes</p>
                    <div className="flex flex-wrap gap-1">
                      {promo.codes.map((c) => (
                        <span key={c} className="text-xs font-mono bg-white border border-gray-200 px-2 py-0.5 rounded">{c}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Audit History */}
            <div className="card">
              <div className="px-5 py-4 border-b border-gray-100">
                <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                  <History size={16} className="text-gray-400" /> Version & Approval History
                </h3>
              </div>
              <div className="px-5 py-3">
                {relatedEvents.length > 0 ? (
                  relatedEvents.map((event) => (
                    <div key={event.id} className="flex items-start gap-3 py-3 border-b border-gray-50 last:border-0">
                      <div className="mt-0.5">
                        {event.action.includes("approved") ? (
                          <CheckCircle2 size={16} className="text-emerald-500" />
                        ) : event.action.includes("submitted") ? (
                          <Send size={16} className="text-purple-500" />
                        ) : (
                          <Clock size={16} className="text-gray-400" />
                        )}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm">
                          <span className="font-semibold">{event.actor}</span>{" "}
                          <span className="text-gray-500">{event.action.replace("promotion.", "")}</span>
                        </p>
                        {event.reason && (
                          <p className="text-xs text-gray-500 mt-0.5">{event.reason}</p>
                        )}
                        <p className="text-xs text-gray-400 mt-1">{formatDateTime(event.timestamp)}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-400 py-4">No audit events recorded yet.</p>
                )}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Schedule */}
            <div className="card p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Calendar size={16} className="text-bass-green" /> Schedule
              </h3>
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-gray-500">Start Date</p>
                  <p className="text-sm font-semibold">{formatDate(promo.startsAt)}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">End Date</p>
                  <p className="text-sm font-semibold">{formatDate(promo.endsAt)}</p>
                </div>
              </div>
            </div>

            {/* Limits */}
            {promo.limits && (
              <div className="card p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                  <Shield size={16} className="text-red-500" /> Limits
                </h3>
                <div className="space-y-3">
                  {promo.limits.totalRedemptions && (
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-500">Total Redemptions</span>
                        <span className="font-semibold">
                          {(promo.limits.usedCount || 0).toLocaleString()} / {promo.limits.totalRedemptions.toLocaleString()}
                        </span>
                      </div>
                      <div className="w-full bg-gray-100 rounded-full h-2">
                        <div
                          className={`rounded-full h-2 transition-all ${
                            ((promo.limits.usedCount || 0) / promo.limits.totalRedemptions) > 0.9
                              ? "bg-red-500"
                              : "bg-bass-gold"
                          }`}
                          style={{
                            width: `${Math.min(100, ((promo.limits.usedCount || 0) / promo.limits.totalRedemptions) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                  {promo.limits.perCustomer && (
                    <div>
                      <p className="text-xs text-gray-500">Per Customer</p>
                      <p className="text-sm font-semibold">{promo.limits.perCustomer} uses</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Created By */}
            <div className="card p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <User size={16} className="text-gray-400" /> Details
              </h3>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="text-xs text-gray-500">Created By</p>
                  <p className="font-medium">{promo.createdBy}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Created</p>
                  <p className="font-medium">{formatDateTime(promo.createdAt)}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Last Updated</p>
                  <p className="font-medium">{formatDateTime(promo.updatedAt)}</p>
                </div>
              </div>
            </div>

            {/* Warnings */}
            {promo.status === "active" && promo.limits?.totalRedemptions && ((promo.limits.usedCount || 0) / promo.limits.totalRedemptions) > 0.8 && (
              <div className="card p-4 border-amber-200 bg-amber-50">
                <div className="flex items-start gap-2">
                  <AlertTriangle size={16} className="text-amber-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-semibold text-amber-800">Approaching Limit</p>
                    <p className="text-xs text-amber-600 mt-0.5">
                      This promotion has used {Math.round(((promo.limits.usedCount || 0) / promo.limits.totalRedemptions) * 100)}% of its total redemption limit. It will be automatically suspended when the limit is reached.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

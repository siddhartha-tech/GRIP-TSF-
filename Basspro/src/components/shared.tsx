"use client";

import React from "react";
import { getStatusColor, getStatusLabel, type PromotionStatus } from "../data/store";

export function StatusBadge({ status }: { status: PromotionStatus }) {
  return (
    <span className={`badge ${getStatusColor(status)}`}>
      <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
        status === "active" ? "bg-emerald-500" :
        status === "in_review" ? "bg-amber-500" :
        status === "approved" ? "bg-blue-500" :
        status === "suspended" ? "bg-red-500" :
        status === "draft" ? "bg-gray-400" :
        "bg-gray-400"
      }`} />
      {getStatusLabel(status)}
    </span>
  );
}

export function SiteBadge({ site }: { site: string }) {
  const siteMap: Record<string, { label: string; color: string }> = {
    "basspro.com": { label: "BP", color: "bg-bass-green/10 text-bass-green" },
    "cabelas.com": { label: "CB", color: "bg-purple-100 text-purple-700" },
    "basspro.ca": { label: "CA", color: "bg-red-100 text-red-700" },
    "mackspw.com": { label: "MW", color: "bg-blue-100 text-blue-700" },
  };
  const s = siteMap[site] || { label: site, color: "bg-gray-100 text-gray-600" };
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold ${s.color}`}>
      {s.label}
    </span>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
        {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function StatCard({
  label,
  value,
  change,
  changeType,
  icon,
}: {
  label: string;
  value: string | number;
  change?: string;
  changeType?: "up" | "down" | "neutral";
  icon?: React.ReactNode;
}) {
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500">{label}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          {change && (
            <p className={`text-xs font-medium mt-2 ${
              changeType === "up" ? "text-emerald-600" :
              changeType === "down" ? "text-red-600" :
              "text-gray-500"
            }`}>
              {changeType === "up" ? "↑ " : changeType === "down" ? "↓ " : ""}{change}
            </p>
          )}
        </div>
        {icon && (
          <div className="p-2.5 rounded-lg bg-gray-50">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4">
      <div className="p-4 rounded-full bg-gray-100 mb-4">{icon}</div>
      <h3 className="text-lg font-semibold text-gray-900 mb-1">{title}</h3>
      <p className="text-sm text-gray-500 text-center max-w-sm mb-4">{description}</p>
      {action}
    </div>
  );
}

export function MoneyText({ cents, currency = "USD" }: { cents: number; currency?: "USD" | "CAD" }) {
  const symbol = currency === "CAD" ? "C$" : "$";
  const formatted = (cents / 100).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return <span>{symbol}{formatted}</span>;
}

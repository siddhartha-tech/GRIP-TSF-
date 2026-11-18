"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, SiteBadge } from "../../components/shared";
import { CODE_POOLS, formatDateTime } from "../../data/store";
import {
  QrCode,
  Plus,
  Download,
  Upload,
  RefreshCw,
  Eye,
  EyeOff,
  Search,
  Copy,
} from "lucide-react";

export default function CodesPage() {
  const [selectedPool, setSelectedPool] = useState<string | null>(null);
  const [showCodes, setShowCodes] = useState(false);

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="Code Management"
          subtitle="Generate, import and export promotion codes"
          actions={
            <div className="flex gap-2">
              <button className="btn-secondary"><Upload size={14} /> Import CSV</button>
              <button className="btn-primary"><Plus size={14} /> Generate Codes</button>
            </div>
          }
        />

        {/* Code Pools */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {CODE_POOLS.map((pool) => (
            <div
              key={pool.id}
              className={`card hover:shadow-md transition-all cursor-pointer ${
                selectedPool === pool.id ? "ring-2 ring-bass-gold" : ""
              }`}
              onClick={() => setSelectedPool(pool.id)}
            >
              <div className="p-5">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className={`p-2 rounded-lg ${
                      pool.type === "public" ? "bg-blue-50" :
                      pool.type === "generated" ? "bg-purple-50" :
                      "bg-amber-50"
                    }`}>
                      <QrCode size={18} className={
                        pool.type === "public" ? "text-blue-600" :
                        pool.type === "generated" ? "text-purple-600" :
                        "text-amber-600"
                      } />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-gray-900">{pool.promotionName}</h3>
                      <p className="text-xs text-gray-400">{pool.id} · {pool.promotionId}</p>
                    </div>
                  </div>
                  <span className={`badge ${
                    pool.type === "public" ? "bg-blue-100 text-blue-700" :
                    pool.type === "generated" ? "bg-purple-100 text-purple-700" :
                    "bg-amber-100 text-amber-700"
                  }`}>
                    {pool.type}
                  </span>
                </div>

                <div className="flex items-center gap-2 mb-3">
                  <SiteBadge site={pool.site} />
                  {pool.prefix && (
                    <span className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded">
                      Prefix: {pool.prefix}
                    </span>
                  )}
                  <span className="text-xs text-gray-400">
                    Length: {pool.length} chars
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-gray-500">Usage</p>
                    <p className="text-sm font-bold">
                      {pool.usedCodes.toLocaleString()} / {pool.totalCodes.toLocaleString()}
                    </p>
                  </div>
                  <div className="w-24">
                    <div className="w-full bg-gray-100 rounded-full h-2">
                      <div
                        className="bg-bass-gold rounded-full h-2"
                        style={{
                          width: `${Math.min(100, (pool.usedCodes / Math.max(1, pool.totalCodes)) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>

                <p className="text-xs text-gray-400 mt-2">
                  Created: {formatDateTime(pool.createdAt)}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Selected Pool Detail */}
        {selectedPool && (
          <div className="card">
            <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-900">
                Code Pool: {CODE_POOLS.find((p) => p.id === selectedPool)?.promotionName}
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowCodes(!showCodes)}
                  className="btn-secondary text-xs py-1.5"
                >
                  {showCodes ? <EyeOff size={12} /> : <Eye size={12} />}
                  {showCodes ? "Mask" : "Reveal"}
                </button>
                <button className="btn-secondary text-xs py-1.5">
                  <Download size={12} /> Export CSV
                </button>
                <button className="btn-secondary text-xs py-1.5">
                  <RefreshCw size={12} /> Refresh
                </button>
              </div>
            </div>
            <div className="p-5">
              {/* Generate form */}
              <div className="p-4 rounded-lg bg-gray-50 border border-gray-100 mb-4">
                <h4 className="text-xs font-semibold text-gray-700 mb-3">Generate New Codes</h4>
                <div className="grid grid-cols-4 gap-3">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Prefix</label>
                    <input type="text" defaultValue="BP-" className="input-field text-xs font-mono" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Length (6–16)</label>
                    <input type="number" defaultValue="8" min="6" max="16" className="input-field text-xs" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Quantity (1–100,000)</label>
                    <input type="number" defaultValue="5000" className="input-field text-xs" />
                  </div>
                  <div className="flex items-end">
                    <button className="btn-primary text-xs w-full">Generate</button>
                  </div>
                </div>
              </div>

              {/* Sample codes table */}
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200">
                    <th className="table-header">Code</th>
                    <th className="table-header">Status</th>
                    <th className="table-header">Used At</th>
                    <th className="table-header">Customer</th>
                    <th className="table-header w-10"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  {[
                    { code: "BP-A3F8K2M1", used: false },
                    { code: "BP-X9R2T5N7", used: true, usedAt: "2026-11-14T10:30:00Z", customer: "cust_***4821" },
                    { code: "BP-M4J7P1Q9", used: true, usedAt: "2026-11-13T15:45:00Z", customer: "cust_***7293" },
                    { code: "BP-K8W3L6V2", used: false },
                    { code: "BP-D5H9N4R8", used: false },
                    { code: "BP-T2F6Y1C3", used: true, usedAt: "2026-11-12T09:20:00Z", customer: "cust_***1056" },
                  ].map((c) => (
                    <tr key={c.code} className="hover:bg-gray-50/50">
                      <td className="table-cell font-mono text-xs">
                        {showCodes ? c.code : c.code.slice(0, 3) + "•••••••"}
                      </td>
                      <td className="table-cell">
                        <span className={`badge ${c.used ? "bg-gray-100 text-gray-600" : "bg-emerald-100 text-emerald-700"}`}>
                          {c.used ? "Used" : "Available"}
                        </span>
                      </td>
                      <td className="table-cell text-xs text-gray-500">
                        {c.usedAt ? formatDateTime(c.usedAt) : "—"}
                      </td>
                      <td className="table-cell text-xs font-mono text-gray-400">
                        {c.customer || "—"}
                      </td>
                      <td className="table-cell">
                        <button className="p-1 text-gray-400 hover:text-gray-600">
                          <Copy size={12} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="px-4 py-3 border-t border-gray-100 text-xs text-gray-400">
                Showing 6 of 50,000 codes · Customer references are masked
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}

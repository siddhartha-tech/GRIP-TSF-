"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader } from "../../components/shared";
import { SCENARIOS, formatCents, formatDateTime } from "../../data/store";
import {
  Play,
  Plus,
  CheckCircle2,
  XCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  Eye,
} from "lucide-react";

export default function ScenariosPage() {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const passed = SCENARIOS.filter((s) => s.passed === true).length;
  const failed = SCENARIOS.filter((s) => s.passed === false).length;
  const total = SCENARIOS.length;

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1200px] mx-auto">
        <PageHeader
          title="Saved Scenarios"
          subtitle="Test carts with expected results — re-run after every change"
          actions={
            <div className="flex gap-2">
              <button className="btn-secondary"><Plus size={14} /> New Scenario</button>
              <button className="btn-primary"><Play size={14} /> Run All</button>
            </div>
          }
        />

        {/* Summary */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="card p-4 text-center">
            <p className="text-2xl font-bold text-emerald-600">{passed}</p>
            <p className="text-xs text-gray-500 font-medium">Passed</p>
          </div>
          <div className="card p-4 text-center">
            <p className="text-2xl font-bold text-red-600">{failed}</p>
            <p className="text-xs text-gray-500 font-medium">Failed</p>
          </div>
          <div className="card p-4 text-center">
            <p className="text-2xl font-bold text-gray-900">{total}</p>
            <p className="text-xs text-gray-500 font-medium">Total</p>
          </div>
        </div>

        {/* Progress bar */}
        <div className="card p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-gray-900">
              {passed} of {total} scenarios passed
            </span>
            <span className="text-xs text-gray-500">
              Last run: {SCENARIOS[0]?.lastRunAt ? formatDateTime(SCENARIOS[0].lastRunAt) : "Never"}
            </span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
            <div className="flex h-full">
              <div
                className="bg-emerald-500 h-full transition-all"
                style={{ width: `${(passed / total) * 100}%` }}
              />
              <div
                className="bg-red-500 h-full transition-all"
                style={{ width: `${(failed / total) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Scenario List */}
        <div className="card overflow-hidden">
          <div className="divide-y divide-gray-100">
            {SCENARIOS.map((scenario) => (
              <div key={scenario.id}>
                <button
                  onClick={() => setExpandedId(expandedId === scenario.id ? null : scenario.id)}
                  className="w-full flex items-center gap-3 p-4 hover:bg-gray-50 transition-colors text-left"
                >
                  {scenario.passed === true ? (
                    <CheckCircle2 size={18} className="text-emerald-500 flex-shrink-0" />
                  ) : scenario.passed === false ? (
                    <XCircle size={18} className="text-red-500 flex-shrink-0" />
                  ) : (
                    <Clock size={18} className="text-gray-300 flex-shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-gray-900">{scenario.name}</span>
                      <span className="text-[10px] font-mono bg-gray-100 px-1.5 py-0.5 rounded text-gray-500">
                        {scenario.id}
                      </span>
                      <span className="text-[10px] font-mono bg-gray-100 px-1.5 py-0.5 rounded text-gray-500">
                        {scenario.templateCode}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                      <span>{scenario.site}</span>
                      <span>Expected: {formatCents(scenario.expectedTotal)}</span>
                      {scenario.lastRunTotal !== undefined && (
                        <span className={scenario.passed ? "text-emerald-600" : "text-red-600"}>
                          Actual: {formatCents(scenario.lastRunTotal)}
                          {!scenario.passed && scenario.lastRunTotal !== undefined && (
                            <span className="ml-1">
                              (Δ {formatCents(Math.abs(scenario.expectedTotal - scenario.lastRunTotal))})
                            </span>
                          )}
                        </span>
                      )}
                    </div>
                  </div>
                  {expandedId === scenario.id ? (
                    <ChevronDown size={16} className="text-gray-400" />
                  ) : (
                    <ChevronRight size={16} className="text-gray-400" />
                  )}
                </button>

                {expandedId === scenario.id && (
                  <div className="px-4 pb-4 ml-8">
                    <div className="p-4 rounded-lg bg-gray-50 border border-gray-100">
                      <h4 className="text-xs font-semibold text-gray-500 mb-2">Cart Contents</h4>
                      <div className="space-y-1.5">
                        {scenario.cart.lines.map((line, idx) => (
                          <div key={idx} className="flex items-center justify-between text-sm">
                            <span>
                              <span className="font-mono text-xs text-gray-400 mr-2">{line.sku}</span>
                              {line.name} × {line.qty}
                            </span>
                            <span className="font-semibold">{formatCents(line.price * line.qty)}</span>
                          </div>
                        ))}
                        {scenario.cart.shipping && (
                          <div className="flex items-center justify-between text-sm pt-1.5 border-t border-gray-200">
                            <span className="text-gray-500">Shipping ({scenario.cart.shipping.method})</span>
                            <span className="font-semibold">{formatCents(scenario.cart.shipping.charge)}</span>
                          </div>
                        )}
                        {scenario.cart.codes && (
                          <div className="flex items-center gap-2 pt-1.5 border-t border-gray-200">
                            <span className="text-xs text-gray-500">Codes:</span>
                            {scenario.cart.codes.map((c) => (
                              <span key={c} className="text-xs font-mono bg-white border border-gray-200 px-2 py-0.5 rounded">{c}</span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex gap-2 mt-3">
                        <button className="btn-secondary text-xs py-1.5">
                          <Play size={12} /> Run This Scenario
                        </button>
                        <button className="btn-secondary text-xs py-1.5">
                          <Eye size={12} /> Open in Simulator
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

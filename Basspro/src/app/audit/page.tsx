"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, SiteBadge } from "../../components/shared";
import { AUDIT_EVENTS, formatDateTime, getRoleLabel } from "../../data/store";
import {
  Search,
  Filter,
  Download,
  ChevronDown,
  ChevronRight,
  Lock,
  Sparkles,
  Zap,
  Clock,
  User,
  FileText,
} from "lucide-react";

export default function AuditPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [actionFilter, setActionFilter] = useState("all");

  const actions = [...new Set(AUDIT_EVENTS.map((e) => e.action))];

  const filtered = AUDIT_EVENTS.filter((e) => {
    if (actionFilter !== "all" && e.action !== actionFilter) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        e.actor.toLowerCase().includes(q) ||
        e.entityName.toLowerCase().includes(q) ||
        e.entityId.toLowerCase().includes(q) ||
        e.action.toLowerCase().includes(q)
      );
    }
    return true;
  });

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1200px] mx-auto">
        <PageHeader
          title="Audit Trail"
          subtitle="Immutable record of every change — who, what, when, before, after, why"
          actions={
            <button className="btn-secondary">
              <Download size={14} /> Export
            </button>
          }
        />

        {/* Info banner */}
        <div className="card p-4 mb-6 bg-blue-50 border-blue-200">
          <div className="flex items-center gap-2">
            <Lock size={14} className="text-blue-500" />
            <p className="text-xs text-blue-700">
              Audit entries are <strong>read-only</strong>. No role can edit or delete an audit entry.
            </p>
          </div>
        </div>

        {/* Filters */}
        <div className="card p-4 mb-6">
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by actor, entity, or action..."
                className="input-field pl-9"
              />
            </div>
            <select
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value)}
              className="input-field w-52"
            >
              <option value="all">All Actions</option>
              {actions.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Events */}
        <div className="card overflow-hidden">
          <div className="divide-y divide-gray-100">
            {filtered.map((event) => (
              <div key={event.id}>
                <button
                  onClick={() => setExpandedId(expandedId === event.id ? null : event.id)}
                  className="w-full flex items-start gap-3 p-4 hover:bg-gray-50 transition-colors text-left"
                >
                  <div className="mt-0.5">
                    {event.automatic ? (
                      <div className="p-1.5 rounded-lg bg-gray-100">
                        <Zap size={14} className="text-gray-500" />
                      </div>
                    ) : event.aiAssisted ? (
                      <div className="p-1.5 rounded-lg bg-purple-100">
                        <Sparkles size={14} className="text-purple-500" />
                      </div>
                    ) : (
                      <div className="p-1.5 rounded-lg bg-blue-50">
                        <User size={14} className="text-blue-500" />
                      </div>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-semibold text-gray-900">{event.actor}</span>
                      <span className="text-xs text-gray-400">({getRoleLabel(event.actorRole)})</span>
                      <span className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded text-gray-500">
                        {event.action}
                      </span>
                      {event.aiAssisted && (
                        <span className="badge bg-purple-100 text-purple-700 text-[10px]">AI-assisted</span>
                      )}
                      {event.automatic && (
                        <span className="badge bg-gray-100 text-gray-600 text-[10px]">Automatic</span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-sm text-gray-700">
                        <span className="font-medium">{event.entityName}</span>
                        <span className="text-gray-400 ml-1.5 text-xs">{event.entityId}</span>
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <Clock size={12} className="text-gray-400" />
                      <span className="text-xs text-gray-400">{formatDateTime(event.timestamp)}</span>
                      <SiteBadge site={event.site} />
                    </div>
                  </div>
                  {expandedId === event.id ? (
                    <ChevronDown size={16} className="text-gray-400 mt-1" />
                  ) : (
                    <ChevronRight size={16} className="text-gray-400 mt-1" />
                  )}
                </button>

                {expandedId === event.id && (
                  <div className="px-4 pb-4 ml-12">
                    <div className="p-4 rounded-lg bg-gray-50 border border-gray-100 space-y-3">
                      {event.reason && (
                        <div>
                          <p className="text-xs font-semibold text-gray-500 mb-1">Reason</p>
                          <p className="text-sm text-gray-700">{event.reason}</p>
                        </div>
                      )}
                      {event.before && (
                        <div>
                          <p className="text-xs font-semibold text-gray-500 mb-1">Before</p>
                          <p className="text-xs font-mono bg-red-50 text-red-700 p-2 rounded">{event.before}</p>
                        </div>
                      )}
                      {event.after && (
                        <div>
                          <p className="text-xs font-semibold text-gray-500 mb-1">After</p>
                          <p className="text-xs font-mono bg-emerald-50 text-emerald-700 p-2 rounded">{event.after}</p>
                        </div>
                      )}
                      <div className="flex items-center gap-4 text-xs text-gray-400 pt-2 border-t border-gray-200">
                        <span>Event ID: {event.id}</span>
                        <span>Entity Type: {event.entityType}</span>
                        <span>Entity ID: {event.entityId}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="px-4 py-3 border-t border-gray-100 text-sm text-gray-500">
            Showing {filtered.length} of {AUDIT_EVENTS.length} events
          </div>
        </div>
      </div>
    </AppShell>
  );
}

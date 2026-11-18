"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, SiteBadge } from "../../components/shared";
import { USERS, getRoleLabel, formatDateTime } from "../../data/store";
import {
  Plus,
  Search,
  Edit,
  Shield,
  User,
  Mail,
  Clock,
  AlertTriangle,
  Lock,
} from "lucide-react";

const ROLE_COLORS: Record<string, string> = {
  super_admin: "bg-red-100 text-red-700",
  site_admin: "bg-amber-100 text-amber-700",
  merchandiser: "bg-blue-100 text-blue-700",
  approver: "bg-emerald-100 text-emerald-700",
  viewer_qa: "bg-purple-100 text-purple-700",
  auditor: "bg-gray-100 text-gray-700",
};

export default function UsersPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedUser, setSelectedUser] = useState<string | null>(null);

  const filtered = USERS.filter((u) => {
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return u.name.toLowerCase().includes(q) || u.email.toLowerCase().includes(q) || u.role.includes(q);
    }
    return true;
  });

  const selected = USERS.find((u) => u.id === selectedUser);

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="User Management"
          subtitle="Manage users, roles and site assignments"
          actions={
            <button className="btn-primary">
              <Plus size={16} /> Add User
            </button>
          }
        />

        {/* Role summary */}
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mb-6">
          {(["super_admin", "site_admin", "merchandiser", "approver", "viewer_qa", "auditor"] as const).map((role) => {
            const count = USERS.filter((u) => u.role === role).length;
            return (
              <div key={role} className="card p-3 text-center">
                <p className="text-xl font-bold text-gray-900">{count}</p>
                <p className={`text-[10px] font-semibold mt-1 badge ${ROLE_COLORS[role]}`}>
                  {getRoleLabel(role)}
                </p>
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* User List */}
          <div className="lg:col-span-2">
            <div className="card">
              <div className="p-4 border-b border-gray-100">
                <div className="relative">
                  <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search users..."
                    className="input-field pl-9"
                  />
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="table-header">User</th>
                      <th className="table-header">Role</th>
                      <th className="table-header">Sites</th>
                      <th className="table-header">Last Login</th>
                      <th className="table-header">Status</th>
                      <th className="table-header w-10"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {filtered.map((user) => (
                      <tr
                        key={user.id}
                        className={`hover:bg-gray-50/50 cursor-pointer transition-colors ${
                          selectedUser === user.id ? "bg-bass-cream/30" : ""
                        }`}
                        onClick={() => setSelectedUser(user.id)}
                      >
                        <td className="table-cell">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-bass-green/10 flex items-center justify-center flex-shrink-0">
                              <span className="text-xs font-bold text-bass-green">
                                {user.name.split(" ").map((n) => n[0]).join("")}
                              </span>
                            </div>
                            <div>
                              <p className="text-sm font-semibold text-gray-900">{user.name}</p>
                              <p className="text-xs text-gray-400">{user.email}</p>
                            </div>
                          </div>
                        </td>
                        <td className="table-cell">
                          <span className={`badge ${ROLE_COLORS[user.role]}`}>
                            {getRoleLabel(user.role)}
                          </span>
                        </td>
                        <td className="table-cell">
                          <div className="flex gap-1 flex-wrap">
                            {user.sites.map((s) => (
                              <SiteBadge key={s} site={s} />
                            ))}
                          </div>
                        </td>
                        <td className="table-cell text-xs text-gray-500">
                          {formatDateTime(user.lastLogin)}
                        </td>
                        <td className="table-cell">
                          <span className={`badge ${user.status === "active" ? "badge-active" : "badge-ended"}`}>
                            {user.status}
                          </span>
                        </td>
                        <td className="table-cell">
                          <button className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-400">
                            <Edit size={14} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* User Detail */}
          <div>
            {selected ? (
              <div className="card p-5 space-y-4">
                <div className="text-center pb-4 border-b border-gray-100">
                  <div className="w-16 h-16 rounded-full bg-bass-green/10 flex items-center justify-center mx-auto mb-3">
                    <span className="text-xl font-bold text-bass-green">
                      {selected.name.split(" ").map((n) => n[0]).join("")}
                    </span>
                  </div>
                  <h3 className="text-lg font-bold text-gray-900">{selected.name}</h3>
                  <p className="text-sm text-gray-500">{selected.email}</p>
                  <span className={`badge mt-2 ${ROLE_COLORS[selected.role]}`}>
                    {getRoleLabel(selected.role)}
                  </span>
                </div>

                <div>
                  <p className="text-xs font-semibold text-gray-500 mb-2">Site Access</p>
                  <div className="flex flex-wrap gap-2">
                    {selected.sites.map((s) => (
                      <div key={s} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-gray-50 border border-gray-100">
                        <SiteBadge site={s} />
                        <span className="text-xs text-gray-600">{s}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-xs font-semibold text-gray-500 mb-2">Permissions</p>
                  <div className="space-y-1.5 text-xs">
                    {selected.role === "super_admin" && (
                      <>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Full access across all sites</div>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> User management</div>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Release publishing</div>
                      </>
                    )}
                    {selected.role === "merchandiser" && (
                      <>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Create & edit promotions</div>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Submit for review</div>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Run simulator</div>
                        <div className="flex items-center gap-2 text-red-400"><Lock size={12} /> Cannot approve own work</div>
                        <div className="flex items-center gap-2 text-red-400"><Lock size={12} /> Cannot pause/archive</div>
                      </>
                    )}
                    {selected.role === "approver" && (
                      <>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Approve/reject promotions</div>
                        <div className="flex items-center gap-2 text-red-400"><Lock size={12} /> Cannot approve own authored versions</div>
                        <div className="flex items-center gap-2 text-red-400"><Lock size={12} /> Cannot create promotions</div>
                      </>
                    )}
                    {selected.role === "site_admin" && (
                      <>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Full lifecycle on own sites</div>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Code management</div>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Release publishing</div>
                      </>
                    )}
                    {selected.role === "viewer_qa" && (
                      <>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> View promotions</div>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Run simulator & scenarios</div>
                        <div className="flex items-center gap-2 text-red-400"><Lock size={12} /> Read-only access</div>
                      </>
                    )}
                    {selected.role === "auditor" && (
                      <>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Read audit trail</div>
                        <div className="flex items-center gap-2 text-emerald-600"><Shield size={12} /> Read decision records</div>
                        <div className="flex items-center gap-2 text-red-400"><Lock size={12} /> Cannot change anything</div>
                      </>
                    )}
                  </div>
                </div>

                <div className="pt-3 border-t border-gray-100">
                  <div className="flex items-center gap-2 text-xs text-gray-400">
                    <Clock size={12} />
                    Last login: {formatDateTime(selected.lastLogin)}
                  </div>
                </div>

                <div className="flex gap-2">
                  <button className="btn-secondary flex-1 text-xs">
                    <Edit size={12} /> Edit Role
                  </button>
                </div>

                {selected.role === "super_admin" && USERS.filter((u) => u.role === "super_admin").length === 1 && (
                  <div className="p-3 rounded-lg bg-amber-50 border border-amber-200">
                    <div className="flex items-start gap-2">
                      <AlertTriangle size={14} className="text-amber-500 mt-0.5 flex-shrink-0" />
                      <p className="text-xs text-amber-700">
                        This is the <strong>only super admin</strong>. Their role cannot be changed to prevent lockout.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="card p-12 text-center">
                <User size={32} className="text-gray-300 mx-auto mb-3" />
                <p className="text-sm text-gray-400">Select a user to view details</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}

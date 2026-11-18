"use client";

import React from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, SiteBadge } from "../../components/shared";
import { CAMPAIGNS, PROMOTIONS, formatDate } from "../../data/store";
import { Plus, FolderOpen, Calendar, User, Tag, ArrowRight } from "lucide-react";
import Link from "next/link";

export default function CampaignsPage() {
  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="Campaigns"
          subtitle="Group promotions into marketing events"
          actions={
            <button className="btn-primary">
              <Plus size={16} /> New Campaign
            </button>
          }
        />

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {CAMPAIGNS.map((campaign) => {
            const campaignPromos = PROMOTIONS.filter((p) => p.campaignId === campaign.id);
            return (
              <div key={campaign.id} className="card hover:shadow-md transition-shadow">
                <div className="p-5">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className={`p-2 rounded-lg ${
                        campaign.status === "active" ? "bg-emerald-50" :
                        campaign.status === "ended" ? "bg-gray-100" :
                        "bg-blue-50"
                      }`}>
                        <FolderOpen size={18} className={
                          campaign.status === "active" ? "text-emerald-600" :
                          campaign.status === "ended" ? "text-gray-400" :
                          "text-blue-600"
                        } />
                      </div>
                      <div>
                        <h3 className="text-sm font-bold text-gray-900">{campaign.name}</h3>
                        <p className="text-xs text-gray-400">{campaign.id}</p>
                      </div>
                    </div>
                    <span className={`badge ${
                      campaign.status === "active" ? "badge-active" :
                      campaign.status === "ended" ? "badge-ended" :
                      "badge-approved"
                    }`}>
                      {campaign.status === "planning" ? "Planning" : campaign.status === "active" ? "Active" : "Ended"}
                    </span>
                  </div>

                  <div className="flex gap-1.5 mb-3">
                    {campaign.sites.map((s) => (
                      <SiteBadge key={s} site={s} />
                    ))}
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="flex items-center gap-1.5 text-gray-500">
                      <Calendar size={12} />
                      <span>{formatDate(campaign.startsAt)} — {formatDate(campaign.endsAt)}</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-gray-500">
                      <User size={12} />
                      <span>{campaign.owner}</span>
                    </div>
                  </div>

                  {campaignPromos.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-gray-100">
                      <p className="text-xs font-semibold text-gray-500 mb-2 flex items-center gap-1">
                        <Tag size={12} /> {campaignPromos.length} Promotions
                      </p>
                      <div className="space-y-1.5">
                        {campaignPromos.slice(0, 3).map((p) => (
                          <Link
                            key={p.id}
                            href={`/promotions/${p.id}`}
                            className="flex items-center justify-between text-xs p-2 rounded bg-gray-50 hover:bg-gray-100 transition-colors"
                          >
                            <span className="font-medium text-gray-700">{p.name}</span>
                            <span className={`badge text-[10px] ${
                              p.status === "active" ? "badge-active" :
                              p.status === "in_review" ? "badge-review" :
                              p.status === "approved" ? "badge-approved" :
                              "badge-draft"
                            }`}>
                              {p.status.replace("_", " ")}
                            </span>
                          </Link>
                        ))}
                        {campaignPromos.length > 3 && (
                          <p className="text-xs text-gray-400 pl-2">+{campaignPromos.length - 3} more</p>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AppShell>
  );
}

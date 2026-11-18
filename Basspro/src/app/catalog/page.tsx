"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, SiteBadge } from "../../components/shared";
import { CATALOG, formatCents } from "../../data/store";
import {
  Search,
  Upload,
  Download,
  Filter,
  Package,
  Tag,
  AlertTriangle,
} from "lucide-react";

export default function CatalogPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [siteFilter, setSiteFilter] = useState("basspro.com");
  const [categoryFilter, setCategoryFilter] = useState("all");

  const filtered = CATALOG.filter((item) => {
    if (item.site !== siteFilter) return false;
    if (categoryFilter !== "all" && !item.categoryPath.startsWith(categoryFilter)) return false;
    if (searchQuery && !item.name.toLowerCase().includes(searchQuery.toLowerCase()) && !item.sku.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const categories = [...new Set(CATALOG.filter((i) => i.site === siteFilter).map((i) => i.categoryPath.split(" > ")[0]))];

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="Catalog Browser"
          subtitle="Browse the pseudo catalog and upload feeds"
          actions={
            <div className="flex gap-2">
              <button className="btn-secondary"><Download size={14} /> Export</button>
              <button className="btn-primary"><Upload size={14} /> Import Feed</button>
            </div>
          }
        />

        {/* Filters */}
        <div className="card p-4 mb-6">
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by SKU or name..."
                className="input-field pl-9"
              />
            </div>
            <select
              value={siteFilter}
              onChange={(e) => setSiteFilter(e.target.value)}
              className="input-field w-44"
            >
              <option value="basspro.com">basspro.com</option>
              <option value="cabelas.com">cabelas.com</option>
              <option value="basspro.ca">basspro.ca</option>
              <option value="mackspw.com">mackspw.com</option>
            </select>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="input-field w-44"
            >
              <option value="all">All Categories</option>
              {categories.map((cat) => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="card p-4 text-center">
            <p className="text-2xl font-bold text-gray-900">{filtered.length}</p>
            <p className="text-xs text-gray-500">Items</p>
          </div>
          <div className="card p-4 text-center">
            <p className="text-2xl font-bold text-emerald-600">
              {filtered.filter((i) => i.promotable).length}
            </p>
            <p className="text-xs text-gray-500">Promotable</p>
          </div>
          <div className="card p-4 text-center">
            <p className="text-2xl font-bold text-amber-600">
              {filtered.filter((i) => i.clearance).length}
            </p>
            <p className="text-xs text-gray-500">Clearance</p>
          </div>
          <div className="card p-4 text-center">
            <p className="text-2xl font-bold text-gray-900">
              {[...new Set(filtered.map((i) => i.brand))].length}
            </p>
            <p className="text-xs text-gray-500">Brands</p>
          </div>
        </div>

        {/* Table */}
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="table-header">SKU</th>
                  <th className="table-header">Name</th>
                  <th className="table-header">Category</th>
                  <th className="table-header">Brand</th>
                  <th className="table-header">Price</th>
                  <th className="table-header">DSCS</th>
                  <th className="table-header">Flags</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {filtered.map((item) => (
                  <tr key={`${item.sku}-${item.site}`} className="hover:bg-gray-50/50">
                    <td className="table-cell font-mono text-xs font-semibold">{item.sku}</td>
                    <td className="table-cell">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded bg-gray-100 flex items-center justify-center flex-shrink-0">
                          <Package size={14} className="text-gray-400" />
                        </div>
                        <span className="text-sm font-medium">{item.name}</span>
                      </div>
                    </td>
                    <td className="table-cell text-xs text-gray-500">{item.categoryPath}</td>
                    <td className="table-cell text-xs">{item.brand}</td>
                    <td className="table-cell font-semibold text-sm">
                      {formatCents(item.price, item.currency)}
                    </td>
                    <td className="table-cell">
                      <span className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded">{item.dscs}</span>
                    </td>
                    <td className="table-cell">
                      <div className="flex gap-1">
                        {item.promotable ? (
                          <span className="badge badge-active text-[10px]">Promotable</span>
                        ) : (
                          <span className="badge badge-suspended text-[10px]">Not Promotable</span>
                        )}
                        {item.clearance && (
                          <span className="badge bg-amber-100 text-amber-700 text-[10px]">Clearance</span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-3 border-t border-gray-100 text-sm text-gray-500">
            Showing {filtered.length} items for {siteFilter}
          </div>
        </div>

        {/* Feed format info */}
        <div className="card p-5 mt-6">
          <h3 className="text-sm font-semibold text-gray-900 mb-3">Feed Format</h3>
          <p className="text-xs text-gray-500 mb-3">
            Upload a CSV with one row per item per site. Required columns:
          </p>
          <div className="flex flex-wrap gap-2">
            {["sku", "site", "name", "price", "currency", "categoryPath", "brand", "dscs", "clearance", "promotable"].map((col) => (
              <span key={col} className="text-xs font-mono bg-gray-100 px-2 py-1 rounded border border-gray-200">
                {col}
              </span>
            ))}
          </div>
          <div className="flex items-center gap-2 mt-3 p-3 rounded-lg bg-amber-50 border border-amber-200">
            <AlertTriangle size={14} className="text-amber-500 flex-shrink-0" />
            <p className="text-xs text-amber-700">
              A feed with a missing price or empty category path will be rejected. All failing rows are named and no rows are saved.
            </p>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
